import numpy as np

class BaseModel:

    def __init__(self) -> None:
        pass

    def update(self, x: np.ndarray, y: np.ndarray):
        pass

    def predict(self):

        pass

    def reset(self):
        pass

