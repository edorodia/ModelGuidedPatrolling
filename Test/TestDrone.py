import sys
sys.path.append('..')

from Environment.PatrollingEnvironment import Drone
import numpy as np
import unittest

class TestDrone(unittest.TestCase):

    navigation_map = np.genfromtxt('../Environment/Maps/map.txt', delimiter=' ')
    position = np.array([20,23])

    def test_influence_side1(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 5)
        drone2 = Drone(self.position, self.navigation_map, 500, 6)
        self.assertEqual(drone1.influence_side, 5)
        self.assertEqual(drone2.influence_side, 7)
    
    def test_influence_side2(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 9)
        drone1.reset()
        self.assertEqual(drone1.position[1], 23)
        self.assertEqual(drone1.position[0], 20)

    def test_influence_side3(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 9)
        drone1.reset()
        expected_mask = np.genfromtxt('mapTest.txt', delimiter=' ')
        self.assertTrue(np.array_equal(drone1._influence_mask(), expected_mask))

    def test_collision_function_inbound_border(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 9)
        self.assertTrue(drone1.collision(np.array([28,1])))
    
    def test_collision_function_outbound(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 9)
        self.assertTrue(drone1.collision(np.array([60,20])))
    
    def test_collision_function_inbound_inlake(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 9)
        self.assertFalse(drone1.collision(np.array([6,11])))

    def test_move_not_collision(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 9)
        drone1.reset()
        #test method response
        self.assertEqual(drone1.move(3,4), "OK")
        #test position change
        self.assertEqual(drone1.position[1], 3)
        self.assertEqual(drone1.position[0], 4)
        #test last waypoint introduced
        self.assertTrue(np.array_equal(drone1.last_waypoints, np.array([[4,3]])))
        #test last waypoint introduced
        self.assertTrue(np.array_equal(drone1.waypoints[-1], np.array([4,3])))
        #test distance added
        self.assertEqual(round(drone1.distance,10), 25.6124969497)

    def test_move_with_collision(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 9)
        drone1.reset()
        #test method response
        self.assertEqual(drone1.move(1,54), "COLLISION")
        #test position not changed
        self.assertEqual(drone1.position[1], 23)
        self.assertEqual(drone1.position[0], 20)
        #test last waypoint introduced
        self.assertTrue(np.array_equal(drone1.last_waypoints, np.array([[20,23]])))
        #test last waypoint introduced
        self.assertTrue(np.array_equal(drone1.waypoints[-1], np.array([20,23])))
        #test distance added
        self.assertEqual(round(drone1.distance,1), 0.0)


if __name__ == '__main__' :
    unittest.main()