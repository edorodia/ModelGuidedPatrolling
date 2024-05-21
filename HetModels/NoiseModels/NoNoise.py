import numpy as np
from BaseNoiseModel import BaseNoiseModel

class NoNoise(BaseNoiseModel):

    # Returns the matrix without editing anything at all #
    def mask(self, matrix: np.ndarray):
        return matrix