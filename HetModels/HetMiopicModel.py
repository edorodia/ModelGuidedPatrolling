import sys
sys.path.append('.')
from HetModels.HetBaseModel import HetBaseModel
import numpy as np

class HetMiopicModel(HetBaseModel):

	# add the influence side to get the data in a correct manner
	def __init__(self, navigation_map: np.ndarray, influence_radius:float, resolution: float = 1.0, dt: float = 0.9) -> None:
		super().__init__()

		self.navigation_map = navigation_map
		self.resolution = resolution

		# Store the positions that are visitable
  		# made of an array of two arrays the first is the x of the positions and the second is the y
		self.visitable_positions = np.array(np.where(self.navigation_map == 1)).T
		# Store the positions that are not visitable
		self.influence_radius = influence_radius
		
		self.model_map = np.zeros_like(self.navigation_map, dtype=np.float32)
		self.dt = dt

	"""
	updates the model data, the update is done on values got by the drone and ASV, the bool flag are introduced to use the model even with
	only one fleet between the two to update values
	"""
	def update(self, from_ASV: bool = False, from_Drone: bool = False, ASV_positions: np.ndarray = None, ASV_values: np.ndarray = None, Drone_positions: np.ndarray = None, Drone_values: np.ndarray = None):  
		
		# If the model is empty, then just add the new data
		if from_ASV :
			# Set all the positions of model_map closer to the position of measurement ASV_positions than influence_radius to ASV_values
			if self.influence_radius != 0:
				for i in range(len(ASV_positions)):
					# Compute all the distances between visitable positions and the positions of the i-th robot
					distances = np.linalg.norm(self.visitable_positions - ASV_positions[i].astype(int), axis=1).astype(int)
					# Get the positions that are closer than influence_radius
					positions = self.visitable_positions[distances <= self.influence_radius]
					# Set the positions to y
					# Sets the positions under the influence radius to the values discovered
					self.model_map[positions[:,0], positions[:,1]] = ASV_values[i]
			else :
				#this is what is done if there is no influence radius every position in the model map gets it corresponding value withouth considering the radius
				self.model_map[ASV_positions[:,0].astype(int), ASV_positions[:,1].astype(int)] = ASV_values
		
		#the drone uses influence_side in another way than ASVs do, every cell is an effective environment read with a different value read
		if from_Drone:
			#this is what is done if there is no influence radius every position in the model map gets it corresponding value withouth considering the radius
			self.model_map[Drone_positions[:,0].astype(int), Drone_positions[:,1].astype(int)] = Drone_values

	def predict(self):

		return self.model_map
	
	def reset(self):

		self.model_map = np.zeros_like(self.navigation_map, dtype=np.float32)
