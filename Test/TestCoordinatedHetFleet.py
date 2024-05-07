import sys
sys.path.append('..')

from Environment.PatrollingEnvironment import Drone, CoordinatedHetFleet
import numpy as np
import unittest

class TestCoordinatedHetFleet(unittest.TestCase):

    navigation_map = np.genfromtxt('../Environment/Maps/map.txt', delimiter=' ')
    # Valid Positions for drone and ASVs #
    position_drone = np.array([[20,23]])
    positions_asv = np.array([[40,45],[10,40]])

    def test_CordHetFleet_initialization(self):
        fleet = CoordinatedHetFleet(
            n_vehicles = 2,
            initial_surface_positions = self.positions_asv,
            navigation_map = self.navigation_map,
            total_max_surface_distance = 1000,
            influence_radius = 2,
            forgetting_factor = 0.01,
            air_forgetting_factor = 0.01,
            initial_air_positions = self.position_drone,
            total_max_air_distance = 1000,
            influence_side = 9,
            camera_fov_angle = 160,
            drone_height = 120,
            blur_data = False,
            drone_idleness_influence = 0.20,
            drone_direct_idleness_influece = False,
            n_drones = 1
        )

        fleet.reset()

        #check position initialization
        self.assertTrue(np.array_equal(fleet.get_ASV_positions(), np.array([[40,45], [10,40]])))
        self.assertTrue(np.array_equal(fleet.get_drone_positions(), np.array([[20,23]])))

    """ test the new idleness maps of air and surface after the move of a drone
    the air_idleness should change while the asv_idleness should stay the same """
    def test_CordHetFleet_move_drone(self):
        fleet = CoordinatedHetFleet(
            n_vehicles = 2,
            initial_surface_positions = self.positions_asv,
            navigation_map = self.navigation_map,
            total_max_surface_distance = 1000,
            influence_radius = 2,
            forgetting_factor = 0.01,
            air_forgetting_factor = 0.01,
            initial_air_positions = self.position_drone,
            total_max_air_distance = 1000,
            influence_side = 9,
            camera_fov_angle = 160,
            drone_height = 120,
            blur_data = False,
            drone_idleness_influence = 0.20,
            drone_direct_idleness_influece = False,
            n_drones = 1
        )

        fleet.reset()

        asv_map = np.copy(fleet.idleness_map)

        fleet.move_Drones([[20,23]])

        zone_drone1 = fleet.idleness_air_map[16:25, 19:28]

        self.assertTrue(np.array_equal(np.asarray(zone_drone1), np.array([[0,0,0,0,0,0,0,0,0],
                                                                         [0,0,0,0,0,0,0,0,0],
                                                                         [0,0,0,0,0,0,0,0,0],
                                                                         [0,0,0,0,0,0,0,0,0],
                                                                         [0,0,0,0,0,0,0,0,0],
                                                                         [0,0,0,0,0,0,0,0,0],
                                                                         [0,0,0,0,0,0,0,0,0],
                                                                         [0,0,0,0,0,0,0,0,0],
                                                                         [0,0,0,0,0,0,0,0,0]])))
        self.assertTrue(np.array_equal(fleet.idleness_map, asv_map))

        

if __name__ == '__main__' :
    unittest.main()