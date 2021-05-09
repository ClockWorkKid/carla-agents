import glob
import os
import sys
import random
import time
import numpy as np

try:
    sys.path.append(glob.glob('carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class CarlaAgent:
    actor_list = []
    client = []
    world = []
    map = []
    BP = []
    vehicle = []
    camera = []
    cameraS = []
    GNSS = []
    collision_sensor = []
    obstacle_sensor = []
    lane_sensor = []
    control = []
    image = []
    imageS = []
    altitude = 0
    latitude = 0
    longitude = 0
    collision = []
    obstacle = []
    lane = []

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

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        self.BP = self.world.get_blueprint_library()

    def spawn_vehicle(self, transform=0):
        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        # BP_vehicle = random.choice(self.BP.filter('vehicle'))
        BP_vehicle = self.BP.find('vehicle.tesla.model3')

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        color = random.choice(BP_vehicle.get_attribute('color').recommended_values)
        BP_vehicle.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        if transform == 0:
            transform = random.choice(self.map.get_spwn_points())

        my_geolocation = self.map.transform_to_geolocation(transform.location)
        self.latitude = my_geolocation.latitude
        self.longitude = my_geolocation.longitude
        self.altitude = my_geolocation.altitude

        # So let's tell the world to spawn the vehicle.
        self.vehicle = self.world.spawn_actor(BP_vehicle, transform)
        # print(transform)
        self.spectator = self.world.get_spectator()

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        self.actor_list.append(self.vehicle)
        print('created %s' % self.vehicle.type_id)

    def attach_camera(self):
        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
            array = array[:, :, :3]  # Take only RGB
            self.image = array

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        BP_camera = self.BP.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(BP_camera, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        print('created %s' % self.camera.type_id)

        # attach camera callback function
        self.camera.listen(lambda image: camera_callback(image))

    # layer 5 => Traffic Light
    # layer 6 => Lanes and marks
    # layer 7 => Roads
    def attach_cameraS(self):
        def semantic_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
            array = array[:, :, :3]  # Take only RGB
            self.imageS = array

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        BP_semantic = self.BP.find('sensor.camera.semantic_segmentation')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.cameraS = self.world.spawn_actor(BP_semantic, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.cameraS)
        print('created %s' % self.cameraS.type_id)

        # attach camera callback function
        self.cameraS.listen(lambda image: semantic_callback(image))

    def attach_GNSS(self):
        def GNSS_callback(gnssData):
            self.altitude = gnssData.altitude
            self.latitude = gnssData.latitude
            self.longitude = gnssData.longitude

        BP_GNSS = self.BP.find('sensor.other.gnss')
        GNSS_transform = carla.Transform(carla.Location(x=0, z=0))
        self.GNSS = self.world.spawn_actor(BP_GNSS, GNSS_transform, attach_to=self.vehicle)
        print('created %s' % self.GNSS.type_id)

        self.GNSS.listen(lambda gnssData: GNSS_callback(gnssData))

    def attach_observer(self):
        transform = carla.Transform(carla.Location(x=0, z=0))
        # --------------
        # Add collision sensor to ego vehicle.
        # --------------
        def col_callback(colli):
            print("Collision detected:\n" + str(colli) + '\n')
            self.collision.append(str(colli))

        col_bp = self.BP.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, transform, attach_to=self.vehicle,
                                    attachment_type=carla.AttachmentType.Rigid)
        self.collision_sensor.listen(lambda colli: col_callback(colli))

        # --------------
        # Add Lane invasion sensor to ego vehicle.
        # --------------
        def lane_callback(lane):
            print("Lane invasion detected:\n" + str(lane) + '\n')
            self.lane.append(str(lane))

        lane_bp = self.BP.find('sensor.other.lane_invasion')
        self.lane_sensor = self.world.spawn_actor(lane_bp, transform, attach_to=self.vehicle,
                                     attachment_type=carla.AttachmentType.Rigid)
        self.lane_sensor.listen(lambda lane: lane_callback(lane))

        # --------------
        # Add Obstacle sensor to ego vehicle.
        # --------------
        def obs_callback(obs):
            print("Obstacle detected:\n" + str(obs) + '\n')
            self.obstacle.append(str(obs))

        obs_bp = self.BP.find('sensor.other.obstacle')
        obs_bp.set_attribute("only_dynamics", str(True))
        self.obstacle_sensor = self.world.spawn_actor(obs_bp, transform, attach_to=self.vehicle,
                                    attachment_type=carla.AttachmentType.Rigid)
        self.obstacle_sensor.listen(lambda obs: obs_callback(obs))

    def autopilot(self, t):
        print("Starting autopilot for " + str(t) + " seconds")
        # Let's put the vehicle to drive around.
        self.vehicle.set_autopilot(True)
        # Drive around for 30 second and then stop autopilot
        time.sleep(t)
        self.vehicle.set_autopilot(False)
        print("Stopping autopilot")

    def attach_controller(self):
        self.vehicle.set_autopilot(False)
        self.control = carla.VehicleControl()

    def find_vehicle(self):
        spec_trans = self.vehicle.get_transform()
        spec_trans.location.x = spec_trans.location.x + 3
        spec_trans.location.z = spec_trans.location.z + 2
        self.spectator.set_transform(spec_trans)

    def terminate(self):
        print('destroying actors')
        self.camera.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print('terminated')