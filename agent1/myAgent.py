import glob
import os
import sys
import random
import time
import numpy as np
import math
from collections import deque

try:
    sys.path.append(glob.glob('carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
STEER_AMT = 1.0
SECONDS_PER_EPISODE = 10.0
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.8

# ****************************************************************************************************************** #
# ************************************************ BASE CARLA AGENT ************************************************ #
# ****************************************************************************************************************** #


class CarlaAgent:
    actor_list = []           # list of all the actors spawned in the server. needs to be cleared on disconnection
    client = None             # client handle to the carla server
    world = None              # carla world
    map = None                # world map
    spectator = None          # spectator on the server window
    BP = None                 # carla actors blueprint library
    vehicle = None            # vehicle actor
    camera = None             # RGB camera actor attached to vehicle
    cameraS = None            # Segmentation camera actor attached to vehicle (ground truth segmentation data)
    GNSS = None               # GPS sensor actor attached to vehicle
    colsensor = None          # collision sensor actor attached to vehicle (needed for reinforcement learning)
    control = None            # control commands
    image = None              # RGB camera data
    imageS = None             # Segmentation camera data
    altitude = 0              # GPS altitude
    latitude = 0              # GPS latitude
    longitude = 0             # GPS longitude
    collision_hist = None     # collision event data

    episode_start = 0         # reinforcement learning episode start timer

    def __init__(self):
        # Connect to server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        time.sleep(4.0)
        print("Establishing Connection to Server")

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spectator = self.world.get_spectator()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        self.BP = self.world.get_blueprint_library()

    def spawn_vehicle(self, BP_vehicle=0, color=0, transform=0):
        # Now let's filter all the blueprints of type 'vehicle' and choose one at random.
        # BP_vehicle = random.choice(self.BP.filter('vehicle'))
        BP_vehicle = self.BP.find('vehicle.tesla.model3')

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if color == 0:
            color = random.choice(BP_vehicle.get_attribute('color').recommended_values)
        BP_vehicle.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        if transform == 0:
            transform = random.choice(self.map.get_spawn_points())

        my_geolocation = self.map.transform_to_geolocation(transform.location)
        self.latitude = my_geolocation.latitude
        self.longitude = my_geolocation.longitude
        self.altitude = my_geolocation.altitude

        # So let's tell the world to spawn the vehicle.
        if self.vehicle is None:
            self.vehicle = self.world.spawn_actor(BP_vehicle, transform)

            # It is important to note that the actors we create won't be destroyed
            # unless we call their "destroy" function. If we fail to call "destroy"
            # they will stay in the simulation even after we quit the Python script.
            # For that reason, we are storing all the actors we create so we can
            # destroy them afterwards.
            self.actor_list.append(self.vehicle)
            print('created %s' % self.vehicle.type_id)
        else:
            print("This agent already has a spawned vehicle")

    def attach_camera(self, height=None, width=None, fov=None):
        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
            array = array[:, :, :3]  # Take only RGB
            self.image = array

        if self.vehicle is None:
            print("Vehicle not spawned")
            return
        # Let's add now a "RGB" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        BP_camera = self.BP.find('sensor.camera.rgb')

        if height is not None and width is not None and fov is not None:
            BP_camera.set_attribute("image_size_x", f"{width}")
            BP_camera.set_attribute("image_size_y", f"{height}")
            BP_camera.set_attribute("fov", f"{fov}")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        if self.camera is None:
            self.camera = self.world.spawn_actor(BP_camera, camera_transform, attach_to=self.vehicle)
            self.actor_list.append(self.camera)
            print('created %s' % self.camera.type_id)
            # attach camera callback function
            self.camera.listen(lambda image: camera_callback(image))
        else:
            print("Camera already exists")

    def attach_cameraS(self):
        def semantic_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
            array = array[:, :, :3]  # Take only RGB
            self.imageS = array

        if self.vehicle is None:
            print("Vehicle not spawned")
            return

        # Let's add now a "semantic" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        BP_semantic = self.BP.find('sensor.camera.semantic_segmentation')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        if self.cameraS is None:
            self.cameraS = self.world.spawn_actor(BP_semantic, camera_transform, attach_to=self.vehicle)
            self.actor_list.append(self.cameraS)
            print('created %s' % self.cameraS.type_id)
            # attach camera callback function
            self.cameraS.listen(lambda image: semantic_callback(image))
        else:
            print("Semantic segmentation camera already exists")

    def attach_GNSS(self):
        def GNSS_callback(gnssData):
            self.altitude = gnssData.altitude
            self.latitude = gnssData.latitude
            self.longitude = gnssData.longitude

        if self.vehicle is None:
            print("Vehicle not spawned")
            return

        BP_GNSS = self.BP.find('sensor.other.gnss')
        GNSS_transform = carla.Transform(carla.Location(x=0, z=0))
        if self.GNSS is None:
            self.GNSS = self.world.spawn_actor(BP_GNSS, GNSS_transform, attach_to=self.vehicle)
            self.actor_list.append(self.GNSS)
            print('created %s' % self.GNSS.type_id)
            self.GNSS.listen(lambda gnssData: GNSS_callback(gnssData))
        else:
            print("GNSS sensor already exists")

    def attach_collision(self):
        def collision_callback(event):
            if self.collision_hist is None:
                self.collision_hist = []
            self.collision_hist.append(event)

        if self.vehicle is None:
            print("Vehicle not spawned")
            return

        BP_col = self.BP.find("sensor.other.collision")
        col_transform = carla.Transform(carla.Location(x=0, z=0))
        if self.colsensor is None:
            self.colsensor = self.world.spawn_actor(BP_col, col_transform, attach_to=self.vehicle)
            self.actor_list.append(self.colsensor)
            print('created %s' % self.colsensor.type_id)
            self.colsensor.listen(lambda event: collision_callback(event))
        else:
            print("Collision sensor exists")

    def autopilot(self, t):
        if self.vehicle is None:
            print("Vehicle not spawned")
            return
        print("Starting autopilot for " + str(t) + " seconds")
        # Let's put the vehicle to drive around.
        self.vehicle.set_autopilot(True)
        # Drive around for 30 second and then stop autopilot
        time.sleep(t)
        self.vehicle.set_autopilot(False)
        print("Stopping autopilot")

    def attach_controller(self):
        if self.vehicle is None:
            print("Vehicle not spawned")
            return
        self.vehicle.set_autopilot(False)
        self.control = carla.VehicleControl()

    def find_vehicle(self):
        if self.vehicle is None:
            print("Vehicle not spawned")
            return
        spec_trans = self.vehicle.get_transform()
        spec_trans.location.x = spec_trans.location.x + 3
        spec_trans.location.z = spec_trans.location.z + 2
        self.spectator.set_transform(spec_trans)

    def terminate(self):
        print('destroying actors')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print('terminated')

    # Reinforcement Learning Reset Method
    def reset(self):
        self.collision_hist = None
        self.actor_list = []

        self.spawn_vehicle()
        self.attach_camera(height=IM_HEIGHT, width=IM_WIDTH, fov=110)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.attach_collision()

        while self.image is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200

        elif kmh < 50:
            done = False
            reward = -1

        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.image, reward, done, None


# ****************************************************************************************************************** #
# *************************************************** DQN Agent **************************************************** #
# ****************************************************************************************************************** #


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def create_model(self):
        model = []

        return model