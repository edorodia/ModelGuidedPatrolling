import numpy as np

class HetBaseModel:

    def __init__(self) -> None:
        pass

    """
    from_drone -> flag to activate when new drone read has occured
    drone_positions -> positions of the drone
    drone_values -> read values by the drone
    exact same thing but for the ASV also    

    the calling default is everything to null or false to avoid errors

    """
    def update(self, from_ASV: bool = False, from_Drone: bool = False, ASV_positions: np.ndarray = None, ASV_values: np.ndarray = None, Drone_positions: np.ndarray = None, Drone_values: np.ndarray = None):
        pass

    def predict(self):

        pass

    def reset(self):
        pass

