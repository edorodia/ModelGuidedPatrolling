import sys
sys.path.append('.')
from Models.BaseModel import BaseModel
import numpy as np

class MiopicModel(BaseModel):

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
		self.x = np.array([])
		self.y = np.array([])

	def update(self, x: np.ndarray, y: np.ndarray, t: np.ndarray = None):
		#x is the array of x values for every position, y is the same for the ys
		#t is the corresponding value found during measurements
  
		# If the model is empty, then just add the new data
		if self.x.size == 0:
			self.x = x
			self.y = y
			if t is not None:
				self.t = t
		else:
			self.x = np.concatenate((self.x, x), axis=0)
			self.y = np.concatenate((self.y, y), axis=0)
			if t is not None:
				self.t = np.concatenate((self.t, t), axis=0)


		# Set all the positions of model_map closer to x than influence_radius to y 

		if self.influence_radius != 0:
			for i in range(len(x)):
				# Compute all the distances between visitable positions and the positions of the i-th robot
				distances = np.linalg.norm(self.visitable_positions - x[i].astype(int), axis=1).astype(int)
				# Get the positions that are closer than influence_radius
				positions = self.visitable_positions[distances <= self.influence_radius]
				# Set the positions to y
				# Sets the positions under the influence radius to the values discovered
				self.model_map[positions[:,0], positions[:,1]] = y[i]
		
		#this is what is done if there is no influence radius every position in the model map gets it corresponding value withouth considering the radius
		self.model_map[self.x[:,0].astype(int), self.x[:,1].astype(int)] = self.y
		

			
	

	def predict(self):

		return self.model_map
	
	def reset(self):

		self.model_map = np.zeros_like(self.navigation_map, dtype=np.float32)

		self.x = np.array([])
		self.y = np.array([])
