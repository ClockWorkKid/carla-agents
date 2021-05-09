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
    image_queue = None
    imageS_queue = None

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

    def __init__(self):
        self.image_queue = Queue()
        self.imageS_queue = Queue()

    def update_image(self):
        def camera_process(buffer):
            image = np.frombuffer(buffer.raw_data, dtype=np.dtype("uint8"))
            image = np.reshape(image, (buffer.height, buffer.width, 4))  # RGBA format
            image = image[:, :, :3]  # Take only RGB
            return image

        self.image = camera_process(self.image_queue.get())
        self.image_queue.queue.clear()

    def update_imageS(self):
        def semantic_process(buffer):
            array = np.frombuffer(buffer.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (buffer.height, buffer.width, 4))  # RGBA format
            array = array[:, :, :3]  # Take only RGB
            return array

        self.imageS = semantic_process(self.imageS_queue.get())
        self.imageS_queue.queue.clear()

    def attach_controller(self):
        self.controller = Controller()
        self.control = carla.VehicleControl()


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
            self.agent.append(new_agent)
        self.world.tick()

    def spawn_vehicle(self, agent, transform=0):
        # BP_vehicle = random.choice(self.BP.filter('vehicle'))
        BP = self.BP.find('vehicle.tesla.model3')

        color = random.choice(BP.get_attribute('color').recommended_values)
        BP.set_attribute('color', color)

        spawned = 0
        while spawned == 0:
            if transform == 0:
                transform = random.choice(self.map.get_spawn_points())

            agent.vehicle = self.world.try_spawn_actor(BP, transform)
            if agent.vehicle is not None:
                spawned = 1

        self.actor_list.append(agent.vehicle)
        print('created %s' % agent.vehicle.type_id)
        car_loc = self.map.transform_to_geolocation(transform.location)
        agent.location = {"altitude": car_loc.altitude, "latitude": car_loc.latitude, "longitude": car_loc.longitude}

    def attach_camera(self, agent):
        BP = self.BP.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        agent.camera = self.world.spawn_actor(BP, camera_transform, attach_to=agent.vehicle)
        self.actor_list.append(agent.camera)
        print('created %s' % agent.camera.type_id)

        agent.camera.listen(agent.image_queue.put)

    def attach_cameraS(self, agent):
        # layer 5 => Traffic Light
        # layer 6 => Lanes and marks
        # layer 7 => Roads
        BP = self.BP.find('sensor.camera.semantic_segmentation')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        agent.cameraS = self.world.spawn_actor(BP, camera_transform, attach_to=agent.vehicle)
        self.actor_list.append(agent.cameraS)
        print('created %s' % agent.cameraS.type_id)

        agent.cameraS.listen(agent.imageS_queue.put)

    def terminate(self):
        print('destroying actors')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print('terminated')

