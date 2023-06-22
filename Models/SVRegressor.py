import sys
sys.path.append('.')
from Models.BaseModel import BaseModel
import numpy as np
from sklearn.svm import SVR

class SVRegressor(BaseModel):

	def __init__(self, navigation_map: np.ndarray) -> None:
		super().__init__()

		self.navigation_map = navigation_map

		# Store the positions that are visitable
		self.visitable_positions = np.array(np.where(self.navigation_map == 1)).T
		# Store the positions that are not visitable
		
		self.model_map = np.zeros_like(self.navigation_map, dtype=np.float32)
		self.x = np.array([])
		self.y = np.array([])

		self.model = SVR(kernel='rbf', gamma='scale', C=1000000.0, tol=1e-5)

	def update(self, x: np.ndarray, y: np.ndarray, t: np.ndarray = None):
		

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

		self.model.fit(self.x, self.y)
		
		self.model_map[self.visitable_positions[:,0], self.visitable_positions[:,1]] = self.model.predict(self.visitable_positions)
		

	def predict(self):

		return self.model_map
	
	def reset(self):

		self.model_map = np.zeros_like(self.navigation_map, dtype=np.float32)

		self.x = np.array([])
		self.y = np.array([])
