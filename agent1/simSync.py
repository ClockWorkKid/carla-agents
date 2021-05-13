import glob
import os
import sys
import random
import time
import numpy as np
from queue import Queue

from Controller import Controller

try:
    sys.path.append(glob.glob('carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 640
IM_HEIGHT = 480

class Vehicle:
    """ Sensors """
    camera = None
    cameraS = None
    GNSS = None
    collision_sensor = None
    obstacle_sensor = None
    lane_sensor = None

    """ Objects """
    vehicle = None

    """ Data """
    location = {}

    image = []
    imageS = []

    collision = []
    lane = []
    obstacle = []

    """ Other """
    waypoint = []
    controller = None
    control = None

    def camera_callback(self, buffer):
        image = np.frombuffer(buffer.raw_data, dtype=np.dtype("uint8"))
        image = np.reshape(image, (buffer.height, buffer.width, 4))  # RGBA format
        image = image[:, :, :3]  # Take only RGB
        self.image = image

    def semantic_callback(self, buffer):
        array = np.frombuffer(buffer.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (buffer.height, buffer.width, 4))  # RGBA format
        array = array[:, :, :3]
        self.imageS = array

    def collision_callback(self, event):
        self.collision.append(event)

    def attach_controller(self):
        self.controller = Controller()
        self.control = carla.VehicleControl()

    def update_waypoint(self, distance=1.0):
        self.waypoint = self.waypoint.next(distance)[0]

    def check_waypoint(self):
        loc = self.vehicle.get_location()
        way = self.waypoint.transform.location

        r = self.vehicle.bounding_box.extent.z
        dist = np.sqrt((loc.x - way.x) ** 2 + (loc.y - way.y) ** 2 + (loc.z - way.z) ** 2)
        return dist, True if dist <= r else False


class SimulatorSynchronous:
    agent = []
    actor_list = []
    spectator = []
    no_agents = 0

    def __init__(self, fps=15, no_agents=1, port=2000):
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(2.0)
        print("Establishing Connection to Server")
        time.sleep(2.0)
        print("Probably connected not sure tho")

        self.world = self.client.get_world()
        self.map = self.world.get_map()

        self.BP = self.world.get_blueprint_library()

        self.spectator = self.world.get_spectator()
        spec_trans = carla.Transform(carla.Location(x=0, y=0, z=250), carla.Rotation(pitch=-90, yaw=180, roll=0))
        self.spectator.set_transform(spec_trans)

        self.settings = self.world.get_settings()

        self.settings.fixed_delta_seconds = 1 / 15  # rendering interval
        self.settings.substepping = True  # physics sub-stepping
        self.settings.max_substep_delta_time = 0.01
        self.settings.max_substeps = 10
        self.settings.synchronous_mode = True  # Enables synchronous mode

        self.world.apply_settings(self.settings)

        self.no_agents = no_agents
        self.world.tick()

    def spawn_agents(self):
        for i in range(self.no_agents):
            new_agent = Vehicle()
            self.spawn_vehicle(new_agent)
            self.attach_camera(new_agent)
            self.attach_collision(new_agent)
            self.agent.append(new_agent)
            self.world.tick()
            print("Vehicle " + str(i) + " spawned")

    def spawn_vehicle(self, agent, transform=0):
        # BP_vehicle = random.choice(self.BP.filter('vehicle'))
        BP = self.BP.find('vehicle.tesla.model3')

        color = random.choice(BP.get_attribute('color').recommended_values)
        BP.set_attribute('color', color)

        spawned = 0
        random.seed(time.time())
        while spawned == 0:
            if transform == 0:
                transform = random.choice(self.map.get_spawn_points())

            agent.vehicle = self.world.try_spawn_actor(BP, transform)
            agent.vehicle.waypoint = self.map.get_waypoint(agent.vehicle.get_location(),
                                                           project_to_road=True,
                                                           lane_type=carla.LaneType.Driving)  # carla.LaneType.Sidewalk
            if agent.vehicle is not None:
                spawned = 1

        self.actor_list.append(agent.vehicle)
        # print('created %s' % agent.vehicle.type_id)
        car_loc = self.map.transform_to_geolocation(transform.location)
        agent.location = {"altitude": car_loc.altitude, "latitude": car_loc.latitude, "longitude": car_loc.longitude}

    def attach_camera(self, agent):
        BP = self.BP.find('sensor.camera.rgb')
        # BP.set_attribute("image_size_x", f"{IM_WIDTH}")
        # BP.set_attribute("image_size_y", f"{IM_HEIGHT}")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        agent.camera = self.world.spawn_actor(BP, camera_transform, attach_to=agent.vehicle)
        self.actor_list.append(agent.camera)
        # print('created %s' % agent.camera.type_id)

        agent.camera.listen(lambda image: agent.camera_callback(image))

    def attach_cameraS(self, agent):
        # layer 5 => Traffic Light
        # layer 6 => Lanes and marks
        # layer 7 => Roads
        BP = self.BP.find('sensor.camera.semantic_segmentation')
        # BP.set_attribute("image_size_x", f"{IM_WIDTH}")
        # BP.set_attribute("image_size_y", f"{IM_HEIGHT}")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        agent.cameraS = self.world.spawn_actor(BP, camera_transform, attach_to=agent.vehicle)
        self.actor_list.append(agent.cameraS)
        # print('created %s' % agent.cameraS.type_id)

        agent.cameraS.listen(lambda image: agent.semantic_callback(image))

    def attach_collision(self, agent):
        BP = self.BP.find("sensor.other.collision")

        transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        agent.collision_sensor = self.world.spawn_actor(BP, transform, attach_to=agent.vehicle)
        self.actor_list.append(agent.collision_sensor)

        agent.collision_sensor.listen(lambda event: agent.collision_callback(event))


    def terminate(self):
        print('destroying actors')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print('terminated')
        self.world.tick()
        self.settings.synchronous_mode = False  # Enables synchronous mode
        self.world.apply_settings(self.settings)

