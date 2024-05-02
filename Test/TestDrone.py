import sys
sys.path.append('..')

from Environment.PatrollingEnvironment import Drone
import numpy as np
import unittest

class TestDrone(unittest.TestCase):

    navigation_map = np.genfromtxt('../Environment/Maps/map.txt', delimiter=' ')
    position = np.array([24,20])

    def test_influence_side1(self):
        drone1 = Drone(self.position, self.navigation_map, 500, 5)
        drone2 = Drone(self.position, self.navigation_map, 500, 6)
        self.assertEqual(drone1.influence_side, 5)
        self.assertEqual(drone2.influence_side, 7)
    
    #def test_influence_side2(self):
    #   drone1 = Drone(self.position, self.navigation_map, 500, 9)

if __name__ == '__main__' :
    unittest.main()