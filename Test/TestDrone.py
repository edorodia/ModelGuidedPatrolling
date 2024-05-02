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

if __name__ == '__main__' :
    unittest.main()