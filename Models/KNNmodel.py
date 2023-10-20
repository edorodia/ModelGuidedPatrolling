import sys
sys.path.append('.')
from Models.BaseModel import BaseModel
import numpy as np
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor

class KNNmodel(BaseModel):

	def __init__(self, navigation_map: np.ndarray, influence_radius:float, resolution: float = 1.0, dt: float = 0.01) -> None:
		super().__init__()

		self.navigation_map = navigation_map
		self.resolution = resolution

		# Store the positions that are visitable
		self.visitable_positions = np.array(np.where(self.navigation_map == 1)).T
		# Store the positions that are not visitable
		self.influence_radius = influence_radius
		self.regressor = RadiusNeighborsRegressor(radius=3, weights='distance', algorithm='auto', n_jobs=1)
		#self.regressor = RadiusNeighborsRegressor(radius=self.influence_radius, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=2)
		
		self.model_map = np.zeros_like(self.navigation_map, dtype=np.float32)
		self.dt = dt

		self.x = np.array([])
		self.y = np.array([])

	def update(self, x: np.ndarray, y: np.ndarray, t: np.ndarray = None):
		

		# If the model is empty, then just add the new data
		if self.x.size == 0:
			self.x = x
			self.y = y
			if t is not None:
				self.t = t*self.dt
		else:
			self.x = np.concatenate((self.x, x), axis=0)
			self.y = np.concatenate((self.y, y), axis=0)
			if t is not None:
				self.t = np.concatenate((self.t, t*self.dt), axis=0)

		# Remove the points close to each other by a distance of 2 if its t is less than 50


		# Fit the model
		if t is not None:
			self.regressor.fit(np.hstack((self.x, np.atleast_2d(self.t).T)), self.y)
			# Update the model map
			y_pred = self.regressor.predict(np.hstack((self.visitable_positions, np.ones(len(self.visitable_positions))[:, None] * self.dt*t.max())))

		else:
			self.regressor.fit(self.x, self.y)
			y_pred = self.regressor.predict(self.visitable_positions)

		y_pred = np.nan_to_num(y_pred, nan=0.0)

		# Update the model map
		self.model_map[self.visitable_positions[:, 0], self.visitable_positions[:, 1]] = np.nan_to_num(y_pred, nan=0.0)

	def predict(self):

		return self.model_map
	
	def reset(self):

		self.model_map = np.zeros_like(self.navigation_map, dtype=np.float32)

		self.x = np.array([])
		self.y = np.array([])

class RKNNmodel(KNNmodel):
	def __init__(self, navigation_map: np.ndarray, influence_radius: float, resolution: float = 1, dt: float = 0.01) -> None:
		super().__init__(navigation_map, influence_radius, resolution, dt)

		self.regressor  = self.regressor = RadiusNeighborsRegressor(radius=self.influence_radius, weights='distance', algorithm='auto', leaf_size=10, p=2, metric='minkowski', n_jobs=2)




