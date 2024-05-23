from HetModels.NoiseModels.BaseNoiseModel import BaseNoiseModel

import numpy as np

class MeanNoise(BaseNoiseModel):

    # Returns the matrix without editing anything at all #
    def mask(self, positions: np.ndarray, values: np.ndarray):
        mean = np.mean(values)
        
        new_values = np.full(len(values), mean)
        
        return positions, new_values