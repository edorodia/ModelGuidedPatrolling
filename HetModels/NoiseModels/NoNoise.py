import sys
from HetModels.NoiseModels.BaseNoiseModel import BaseNoiseModel

import numpy as np

class NoNoise(BaseNoiseModel):

    # Returns the matrix without editing anything at all #
    def mask(self, positions: np.ndarray, values: np.ndarray):
        return positions, values