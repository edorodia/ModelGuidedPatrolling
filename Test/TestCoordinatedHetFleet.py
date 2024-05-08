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

        asv_map =   np.copy(fleet.idleness_map)

        drone_map = np.copy(fleet.idleness_air_map)
        drone_map = drone_map * self.navigation_map
        drone_map[16:25, 19:28] = 0

        fleet.move_Drones([[20,23]])

        self.assertTrue(np.array_equal(fleet.idleness_air_map, drone_map))
        self.assertTrue(np.array_equal(fleet.idleness_map, asv_map))

    """ test the new idleness maps of air and surface after the move of the ASVs
    both the two idleness maps should change """
    def test_CordHetFleet_move_ASV(self):
        fleet = CoordinatedHetFleet(
            n_vehicles = 2,
            initial_surface_positions = np.array([[11,39],[41,44]]),
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

        # the ASV_map that it's expected
        asv_map =   np.copy(fleet.idleness_map)
        asv_map = asv_map * self.navigation_map
        asv_map[8,40]             =0
        asv_map[9,39:42]          =0
        asv_map[10,38:43]         =0
        asv_map[11,39:42]         =0
        asv_map[12,40]            =0

        asv_map[38,45]            =0
        asv_map[39,44:47]         =0
        asv_map[40,43:48]         =0
        asv_map[41,44:47]         =0
        asv_map[42,45]            =0

        # the AIR_map that it's expected
        drone_map = np.copy(fleet.idleness_air_map)
        drone_map = drone_map * self.navigation_map
        drone_map[8,40]             =0
        drone_map[9,39:42]          =0
        drone_map[10,38:43]         =0
        drone_map[11,39:42]         =0
        drone_map[12,40]            =0

        drone_map[38,45]            =0
        drone_map[39,44:47]         =0
        drone_map[40,43:48]         =0
        drone_map[41,44:47]         =0
        drone_map[42,45]            =0

        
        movements = [
                        {'length': 1, 'angle': np.radians(135)},  # Movement for vehicle 0
                        {'length': 1, 'angle': np.radians(135)}   # Movement for vehicle 1
                    ]
        
        fleet.move_ASVs(movements)

        self.assertTrue(np.array_equal(fleet.idleness_air_map, drone_map))
        self.assertTrue(np.array_equal(fleet.idleness_map, asv_map))
        
    def test_render(self):
        fleet = CoordinatedHetFleet(
            n_vehicles = 2,
            initial_surface_positions = np.array([[11,39],[41,44]]),
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

        movements = [
                        {'length': 1, 'angle': np.radians(135)},  # Movement for vehicle 0
                        {'length': 1, 'angle': np.radians(135)}   # Movement for vehicle 1
                    ]

        fleet.move_ASVs(movements)

        fleet.move_Drones([[20,23]])

        fleet.render()

        fleet.move_ASVs(movements)

        fleet.move_Drones([[40,23]])

        fleet.render()

        #input("premi un tasto per chiudere")



if __name__ == '__main__' :
    unittest.main()