import sys
sys.path.append('..')

from HetModels.NoiseModels.FishEyeNoiseApproximator import FishEyeNoiseApproximator
import numpy as np
import unittest

class TestNoiseModels(unittest.TestCase):

    def testFishEye(self):
        positions = np.array([[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
                              [4,4],[4,5],[4,6],[4,7],[4,8],[4,9],[4,10],
                              [5,4],[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],
                              [6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],
                              [7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],
                              [8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[8,10],
                              [9,4],[9,5],[9,6],[9,7],[9,8],[9,9],[9,10]])
        
        values =    np.array([2,1,10,20,10,2,3,
                              3,4,10,30,10,4,5,
                              5,6,6,7,8,7,8,
                              7,8,9,10,11,9,10,
                              9,10,12,13,14,11,12,
                              4,5,8,9,10,3,4,
                              6,7,11,12,13,5,6])
        
        final_positions =   np.array([[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
                                      [4,4],[4,5],[4,6],[4,7],[4,8],[4,9],[4,10],
                                      [5,4],[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],
                                      [6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],
                                      [7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],
                                      [8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[8,10],
                                      [9,4],[9,5],[9,6],[9,7],[9,8],[9,9],[9,10]])
        final_values =      np.array([2.5,2.5,15.0,15.0,15.0,3.5,3.5,
                                      2.5,2.5,15.0,15.0,15.0,3.5,3.5,
                                      7.5,7.5,6.0,7.0,8.0,9.5,9.5,
                                      7.5,7.5,9.0,10.0,11.0,9.5,9.5,
                                      7.5,7.5,12.0,13.0,14.0,9.5,9.5,
                                      5.5,5.5,10.5,10.5,10.5,4.5,4.5,
                                      5.5,5.5,10.5,10.5,10.5,4.5,4.5])
        
        model = FishEyeNoiseApproximator(influence_side=7, internal_true_to_cell_side=3)
        positions_returned, values_returned = model.mask(positions=positions, values=values)
        self.assertTrue(np.array_equal(final_positions, positions_returned))
        self.assertTrue(np.array_equal(final_values, values_returned))
        
        

if __name__ == '__main__' :
    unittest.main()