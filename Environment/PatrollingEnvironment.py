import numpy as np
import matplotlib.pyplot as plt
from Environment.GroundTruths.AlgaeBloomGroundTruth import algae_bloom
from Environment.GroundTruths.ShekelGroundTruth import shekel
from gym import spaces
from Models.KNNmodel import KNNmodel, RKNNmodel
from Models.MiopicModel import MiopicModel
from Models.GaussianProcessModel import GaussianProcessModel
from Models.PolinomialRegressor import PolinomialRegressor
from Models.SVRegressor import SVRegressor
from Models.UnetModel import UnetDeepModel, VAEUnetDeepModel, benchmark_2_vae_path
from sklearn.metrics import mean_squared_error, r2_score
from PathPlanners.dijkstra import Dijkstra


class Vehicle:
	
	def __init__(self,
	             initial_positions: np.ndarray,
	             navigation_map: np.ndarray,
	             total_max_distance: float,
	             influence_radius: float):
		
		# Copy the initial positions #
		self.initial_positions = np.atleast_2d(initial_positions).copy()
		self.navigation_map = navigation_map.copy()
		self.local_planner = Dijkstra(obstacle_map=1 - self.navigation_map, robot_radius=0.0)
		
		# Initialize positions
		self.position = None
		self.total_max_distance = total_max_distance
		self.distance = 0.0
		self.waypoints = []
		self.last_waypoints = []
		self.steps = 0
		self.influence_radius = influence_radius
		
		self.influence_mask = np.zeros_like(self.navigation_map)
	
	def reset(self):
		
		# Get a random position from the initial  positions
		self.position = self.initial_positions[np.random.randint(0, len(self.initial_positions))]
		
		# Reset the distance
		self.distance = 0.0
		
		# Reset the path
		self.waypoints = [self.position]
		self.last_waypoints = [self.position]
		
		self.influence_mask = self._influence_mask()
		
		self.steps = 0
	
	def _influence_mask(self):
		""" Create a 0 matrix with the size of the navigation map and set to 1 a circle centered in the position of
		the vehicle of radius self.influence_radius"""
		
		influence_mask = np.zeros_like(self.navigation_map)
		
		# Set the influence mask to 1 in the circle centered in the position of the vehicle #
		
		# Compute the coordinates of the circle #
		x, y = np.meshgrid(np.arange(0, influence_mask.shape[1]), np.arange(0, influence_mask.shape[0]))
		x = x - self.position[1].astype(int)
		y = y - self.position[0].astype(int)
		
		# Compute the distance from the center #
		distance = np.sqrt(x ** 2 + y ** 2)
		
		# Set to 1 the points in the circle #
		influence_mask[distance <= self.influence_radius] = 1
		
		return influence_mask
	
	def move(self, length, angle, action_type='discrete'):
		# Take a step in given direction #
		
		self.steps += 1
		
		direction = np.array([np.cos(angle), np.sin(angle)])
		if action_type == 'discrete':
			direction = np.round(direction).astype(int)
		
		self.last_waypoints = []
		
		number_of_minimoves = np.round(length).astype(int)
		minimove_length = 1
		
		original_position = self.position.copy()
		next_target_position = self.position.copy()
		
		# Iterate for minimoves #
		for minimove in range(1, number_of_minimoves + 1):
			
			# Check if the next_target_position is visitable #
			next_target_position = (original_position + minimove * minimove_length * direction).astype(int)
			
			collision = self.collision(next_target_position)
			
			if collision:
				
				last_truncated_position = self.position.copy()
				
				self.waypoints.append(original_position + (minimove - 1 / number_of_minimoves) * direction)
				self.last_waypoints.append(self.waypoints[-1])
				self.influence_mask = self._influence_mask()
				self.position = self.waypoints[-1]
				self.last_waypoints = np.asarray(self.last_waypoints)
				return "COLLISION"
			
			else:
				
				# Compute distance #
				self.distance += np.linalg.norm(next_target_position - self.position)
				
				# Copy the next position #
				self.position = next_target_position.copy()
				self.waypoints.append(self.position)
				self.last_waypoints.append(self.position)
				self.influence_mask = self._influence_mask()
		
		self.last_waypoints = np.asarray(self.last_waypoints)
		
		return "OK"
	
	def collision(self, position):
		# Check if the vehicle has collided #
		c_position = position.copy().astype(int)
		
		in_bound = (0 <= int(c_position[0]) < self.navigation_map.shape[0] and 0 <= int(c_position[1]) <
		            self.navigation_map.shape[1])
		
		if not in_bound:
			return True
		else:
			return not self.navigation_map[c_position[0], c_position[1]].astype(bool)
	
	def move_to_position(self, new_position):
		
		self.steps += 1
		
		if self.collision(new_position):
			return "COLLISION"
		
		else:
			# Compute the path using dijkstra #
			path = self.local_planner.planning(self.position, new_position)
			self.position = new_position.copy().astype(int)
			self.waypoints.extend([pos for pos in path])
			self.last_waypoints = np.asarray([pos for pos in path])
			self.influence_mask = self._influence_mask()
			self.distance += np.linalg.norm(self.position - self.waypoints[-1])
			
			return "OK"


class CoordinatedFleet:
	
	def __init__(self,
	             n_vehicles: int,
	             initial_positions: np.ndarray,
	             navigation_map: np.ndarray,
	             total_max_distance: float,
	             influence_radius: float,
	             max_num_of_steps: float,
	             ):
		
		self.navigation_map = navigation_map
		self.initial_positions = initial_positions
		self.total_max_distance = total_max_distance
		self.n_vehicles = n_vehicles
		self.max_num_of_steps = max_num_of_steps
		
		self.idleness_map = np.ones_like(self.navigation_map)
		self.idleness_map_ = np.ones_like(self.navigation_map)
		self.changes_in_idleness = np.zeros_like(self.navigation_map)
		
		# Create the fleet #
		self.vehicles = [Vehicle(initial_positions=self.initial_positions[i],
		                         navigation_map=navigation_map,
		                         total_max_distance=total_max_distance,
		                         influence_radius=influence_radius) for i in range(n_vehicles)]
		
		self.vehicles_ids = set(range(n_vehicles))
		
		self.fig = None
	
	def reset(self):
		
		# Reset the vehicles #
		self.vehicles_ids = set(range(self.n_vehicles))
		self.visited_map = np.zeros_like(self.navigation_map)
		for vehicle in self.vehicles:
			vehicle.reset()
			self.visited_map[vehicle.position[0].astype(int), vehicle.position[1].astype(int)] = 1
		
		self.idleness_map = np.ones_like(self.navigation_map)
		self.idleness_map_ = np.ones_like(self.navigation_map)
	
	def move(self, movements, action_type='discrete'):
		
		# Set the flags for inactive agents #
		remove_ids = []
		for vehicle_id in self.vehicles_ids:
			if self.vehicles[vehicle_id].distance > self.total_max_distance:
				remove_ids.append(vehicle_id)
		
		# Remove the inactive agents #
		for vehicle_id in remove_ids:
			self.vehicles_ids.remove(vehicle_id)
		
		# Move the vehicles #
		for vehicle_id in self.vehicles_ids:
			
			if action_type == 'discrete':
				self.vehicles[vehicle_id].move(movements[vehicle_id]['length'], movements[vehicle_id]['angle'])
			elif action_type == 'next_position':
				self.vehicles[vehicle_id].move_to_position(movements[vehicle_id])
			
			# Update the visited map #
			self.visited_map[
				self.vehicles[vehicle_id].last_waypoints[:, 0].astype(int),
				self.vehicles[vehicle_id].last_waypoints[:, 1].astype(int)] = 1
		
		self.update_idleness_map()
	
	def get_vehicle_position_map(self, observer: int):
		""" Get the map with the observed position of the agent """
		
		position_map = np.zeros_like(self.navigation_map)
		position_map[
			self.vehicles[observer].position[0].astype(int), self.vehicles[observer].position[1].astype(int)] = 1
		
		return position_map
	
	def get_vehicle_trajectory_map(self, observer: int):
		
		position_map = np.zeros_like(self.navigation_map)
		
		waypoints = np.asarray(self.vehicles[observer].waypoints)
		
		position_map[waypoints[-10:, 0].astype(int), waypoints[-10:, 1].astype(int)] = np.linspace(0, 1,
		                                                                                           len(waypoints[
		                                                                                               -10:]),
		                                                                                           endpoint=True)
		
		return position_map
	
	def get_fleet_position_map(self, observers: np.ndarray):
		""" Get the map with the observed position of the agent """
		
		observers = np.atleast_1d(observers)
		position_map = np.zeros_like(self.navigation_map)
		
		for agent_id in self.vehicles_ids:
			
			if agent_id in observers:
				continue
			
			position_map[
				self.vehicles[agent_id].position[0].astype(int), self.vehicles[agent_id].position[1].astype(int)] = 1
		
		return position_map
	
	def get_positions(self):
		""" Return an array with the XY positions of the active agents """
		return np.array([self.vehicles[vehicle_id].position for vehicle_id in self.vehicles_ids])
	
	def render(self):
		
		if self.fig is None:
			
			self.fig, self.ax = plt.subplots(1, 1)
			
			self.ax_pos = []
			
			self.ax.imshow(self.navigation_map, cmap='gray', vmin=0, vmax=1)
			
			# Plot all vehicles' positions #
			for vehicle in self.vehicles:
				self.ax_pos.append(self.ax.plot(vehicle.position[1], vehicle.position[0], 'ro')[0])
			
			plt.show(block=False)
		else:
			
			# Update the positions #
			for vehicle_id in self.vehicles_ids:
				self.ax_pos[vehicle_id].set_xdata(self.vehicles[vehicle_id].position[1])
				self.ax_pos[vehicle_id].set_ydata(self.vehicles[vehicle_id].position[0])
		
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		plt.pause(0.1)
	
	def redundancy_mask(self):
		
		# Sum all the influence masks #
		return np.array([self.vehicles[i].influence_mask for i in self.vehicles_ids]).sum(axis=0)
	
	def changes_idleness(self):
		
		# Compute the changes in idleness #
		net_change = np.abs(self.idleness_map - self.idleness_map_)
		
		# Compute the changes for every vehicle by vehicle mask multiplication #
		return {i: np.sum(self.vehicles[i].influence_mask * (net_change / np.sum(self.redundancy_mask()))) for i in
		        self.vehicles_ids}
	
	def update_idleness_map(self):
		""" Update the idleness map """
		
		# Copy the previous idleness map #
		self.idleness_map_ = self.idleness_map.copy()
		
		# Update the idleness map #
		self.idleness_map += 1 / self.max_num_of_steps  # Increment the idleness map everywhere
		
		self.idleness_map = np.clip(self.idleness_map, 0, 1)  # Clip the idleness map
		
		# Reset the idleness map in the vehicles influence area #
		for vehicle_id in self.vehicles_ids:
			self.idleness_map[np.where(self.vehicles[vehicle_id].influence_mask != 0)] = 0
		
		self.idleness_map = self.idleness_map * self.navigation_map
	
	def get_last_waypoints(self):
		""" Return the last waypoints of the vehicles """
		return np.vstack([vehicle.last_waypoints for vehicle in self.vehicles])


class DiscreteModelBasedPatrolling:
	
	def __init__(self,
	             n_agents: int,
	             navigation_map: np.ndarray,
	             initial_positions: np.ndarray,
	             model_based: bool,
	             movement_length: int,
	             resolution: int,
	             max_distance: float,
	             influence_radius: float,
	             forgetting_factor: float,
	             reward_type='weighted_importance',
	             reward_weights=(1, 1),
	             benchmark: str = 'algae_bloom',
	             model: str = 'miopic',
	             dynamic: bool = True,
	             seed: int = 0,
	             random_gt: bool = True,
	             int_observation: bool = False,
	             min_information_importance: float = 1.0,
	             ):
		
		""" Copy attributes """
		
		self.min_information_importance = min_information_importance
		self.eval = False
		np.random.seed(seed)  # Initialize the random seed
		
		self.n_agents = n_agents
		self.number_of_agents = n_agents
		self.navigation_map = navigation_map
		self.scenario_map = navigation_map
		self.initial_positions = initial_positions
		self.model_based = model_based
		self.move_length = movement_length
		self.movement_length = movement_length
		self.resolution = resolution
		self.max_distance = max_distance
		self.influence_radius = influence_radius
		self.seed = seed
		self.dynamic = dynamic
		self.random_gt = random_gt
		
		self.true_reward = {}
		
		self.int_observation = int_observation
		
		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1))
		
		self.lambda_W = reward_weights[0]
		self.lambda_I = reward_weights[1]
		
		self.max_num_steps = (forgetting_factor * self.max_distance) // self.move_length
		
		self.max_agent_steps = self.max_distance // self.move_length + 1
		
		""" Create the fleet """
		self.fleet = CoordinatedFleet(n_vehicles=self.n_agents,
		                              initial_positions=self.initial_positions,
		                              navigation_map=self.navigation_map,
		                              total_max_distance=self.max_distance,
		                              influence_radius=self.influence_radius,
		                              max_num_of_steps=self.max_num_steps, )
		
		""" Create the observation space """
		
		self.action_space = spaces.Discrete(8)
		
		self.observation_space = spaces.Box(low=0, high=1,
		                                    shape=(4, self.navigation_map.shape[0], self.navigation_map.shape[1]),
		                                    dtype=np.uint8 if self.int_observation else np.float32)
		
		""" Create the reward function """
		self.reward_type = reward_type
		
		self.model_str = model
		
		""" Create the model """
		if model == 'knn':
			self.model = KNNmodel(navigation_map=self.navigation_map,
			                      resolution=self.resolution,
			                      influence_radius=self.influence_radius,
			                      dt=0.01)
		elif model == 'miopic':
			self.model = MiopicModel(navigation_map=self.navigation_map,
			                         resolution=self.resolution,
			                         influence_radius=self.influence_radius,
			                         dt=0.7)
		elif model == 'rknn':
			self.model = RKNNmodel(navigation_map=self.navigation_map,
			                       resolution=self.resolution,
			                       influence_radius=self.influence_radius, dt=0.01)
		elif model == 'vaeUnet':
			self.model = VAEUnetDeepModel(navigation_map=self.navigation_map,
			                              model_path=benchmark_2_vae_path[benchmark],
			                              resolution=self.resolution,
			                              influence_radius=0.0, dt=0.01, N_imagined=1)
		elif model == 'gp':
			self.model = GaussianProcessModel(navigation_map=self.navigation_map,
			                                  resolution=self.resolution,
			                                  influence_radius=self.influence_radius, dt=0.01)
		elif model == 'poly':
			self.model = PolinomialRegressor(navigation_map=self.navigation_map,
			                                 degree=4)
		elif model == 'svr':
			self.model = SVRegressor(navigation_map=self.navigation_map)
		else:
			raise ValueError('Unknown model')
		
		self.fig = None
		
		""" Create the benchmark """
		if benchmark == 'shekel':
			self.ground_truth = shekel(self.navigation_map, max_number_of_peaks=6, seed=self.seed, dt=0.1)
		elif benchmark == 'algae_bloom':
			self.ground_truth = algae_bloom(self.navigation_map, dt=0.05, seed=self.seed)
		else:
			raise ValueError('Unknown benchmark')
		
		self.ground_truth.reset()
	
	def get_positions(self):
		
		return self.fleet.get_positions()
	
	def get_all_positions(self):
		
		return np.asarray([vehicle.position for vehicle in self.fleet.vehicles])
	
	def get_positions_dict(self):
		
		return {agent_id: position for agent_id, position in zip(self.fleet.vehicles_ids, self.fleet.get_positions())}
	
	def reset(self):
		
		""" Reset the fleet """
		self.fleet.reset()
		
		""" Reset the ground truth """
		if self.random_gt:
			self.ground_truth.reset()
		
		""" Reset the model """
		self.model.reset()
		self.previous_model = self.model.predict()
		
		self.info = {}
		self.true_reward = {}
		
		self.steps = 0
		
		return self.get_observations()
	
	def action_to_movement(self, action: int):
		""" Convert the action to a movement order """
		
		return {'length': self.move_length, 'angle': action * 2.0 * np.pi / 8.0}
	
	def random_action(self):
		""" Return a random action """
		return self.action_to_movement(np.random.randint(8))
	
	def step(self, actions: dict, action_type='discrete'):
		
		self.steps += 1
		
		if action_type == 'discrete':
			movements_orders = {key: self.action_to_movement(action) for key, action in actions.items()}
		else:
			movements_orders = actions
		
		# Move the fleet #
		self.fleet.move(movements_orders, action_type=action_type)
		
		# Update the model #
		self.update_model()
		
		# Get the observations #
		observations = self.get_observations()
		
		# Get the rewards #
		rewards = self.get_rewards()
		
		# Get the done flag #
		done = self.get_done()
		
		if self.dynamic:
			self.ground_truth.step()
		
		if self.eval:
			self.info = self.get_info()
		
		return observations, rewards, done, self.info
	
	def update_model(self):
		""" Update the model """
		
		# Obtain all the new positions of the agents #
		# positions = self.fleet.get_positions()
		
		sample_positions = self.fleet.get_last_waypoints()
		
		# Obtain the values of the ground truth in the new positions #
		values = self.ground_truth.read(sample_positions)
		
		self.previous_model = self.model.predict().copy()
		
		# Update the model #
		
		if self.model_str == 'deepUnet' or self.model_str == 'vaeUnet':
			self.model.update(sample_positions, values, self.fleet.visited_map)
		else:
			self.model.update(sample_positions, values)
	
	def get_observations(self):
		""" Observation function. The observation is composed by:
		1) Navigation map with obstacles
		2) Positions of i agent
		3) Positions of the other agents
		4) Idleness
		5) Model
		"""
		
		# The observation is a dictionary of every active agent's observation #
		
		observations = {}
		
		for vehicle_id in self.fleet.vehicles_ids:
			
			if self.int_observation:
				
				observations[vehicle_id] = np.concatenate((
						# (255 * self.navigation_map[np.newaxis]).astype(np.uint8),
						(255 * self.fleet.get_vehicle_trajectory_map(observer=vehicle_id)[np.newaxis]).astype(
								np.uint8),
						(255 * self.fleet.get_fleet_position_map(observers=vehicle_id)[np.newaxis]).astype(np.uint8),
						(255 * self.fleet.idleness_map[np.newaxis]).astype(np.uint8),
						(255 * self.model.predict()[np.newaxis]).astype(np.uint8),
				), axis=0)
			
			else:
				
				observations[vehicle_id] = np.concatenate((
						# self.navigation_map[np.newaxis],
						self.fleet.get_vehicle_trajectory_map(observer=vehicle_id)[np.newaxis],
						self.fleet.get_fleet_position_map(observers=vehicle_id)[np.newaxis],
						self.fleet.idleness_map[np.newaxis],
						self.model.predict()[np.newaxis],
				), axis=0)
		
		self.observations = observations
		
		return self.observations
	
	def get_rewards(self):
		""" The reward is selected dependign on the reward type """
		
		reward = {}
		
		self.true_reward = {}
		
		if self.reward_type == 'weighted_idleness':
			
			# Compute the reward as the local changes for the agent. #
			
			W = self.fleet.changes_idleness()  # Compute the idleness changes #
			
			for agent_id in self.fleet.vehicles_ids:
				# The reward is the sum of the local changes in the agent's influence area + W #
				
				information_gain = np.sum(self.fleet.vehicles[agent_id].influence_mask * (
						self.model.predict() / np.sum(self.fleet.redundancy_mask())))
				
				true_information_gain = np.sum(self.fleet.vehicles[agent_id].influence_mask * (
						self.ground_truth.read() / np.sum(self.fleet.redundancy_mask())))
				
				reward[agent_id] = W[agent_id] * (information_gain + self.min_information_importance) * 100.0
				self.true_reward[agent_id] = W[agent_id] * (
						true_information_gain + self.min_information_importance) * 100.0
		
		else:
			raise NotImplementedError('Unknown reward type')
		
		return reward
	
	def get_done(self):
		""" End the episode when the distance is greater than the max distance """
		
		done = {agent_id: self.fleet.vehicles[agent_id].distance > self.max_distance or self.fleet.vehicles[agent_id].steps > self.max_agent_steps for agent_id in
		        self.fleet.vehicles_ids}
		
		return done
	
	def render(self):
		
		if self.fig is None:
			
			self.fig, self.axs = plt.subplots(3, 2)
			self.axs = self.axs.flatten()
			
			# Plot the navigation map #
			self.d0 = self.axs[0].imshow(self.navigation_map, cmap='gray', vmin=0, vmax=1)
			# self.d0 = self.axs[0].imshow(self.ground_truth.read(), cmap='gray', vmin=0, vmax=1)
			self.axs[0].set_title('Navigation map')
			self.d1 = self.axs[1].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][0], cmap='gray', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[1].set_title('Agent Position')
			self.d2 = self.axs[2].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][1], cmap='gray', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[2].set_title('Fleet Position')
			self.d3 = self.axs[3].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][2], cmap='jet', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[3].set_title('Idleness')
			self.d4 = self.axs[4].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][3], cmap='jet', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[4].set_title('Model')
			self.d5 = self.axs[5].imshow(self.ground_truth.read(), cmap='jet', vmin=0, vmax=1)
			plt.colorbar(self.d0, ax=self.axs[0])
		
		else:
			
			self.d0.set_data(self.navigation_map)
			# self.d0.set_data(self.ground_truth.read())
			self.d1.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][0])
			self.d2.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][1])
			self.d3.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][2])
			self.d4.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][3])
			self.d5.set_data(self.ground_truth.read())
		
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		plt.pause(0.01)
	
	def get_info(self):
		""" This method returns the info of the step """
		
		y_real = self.ground_truth.read()[self.visitable_positions[:, 0], self.visitable_positions[:, 1]]
		y_pred = self.model.predict()[self.visitable_positions[:, 0], self.visitable_positions[:, 1]]
		
		mse_error = np.mean((y_real - y_pred) ** 2)
		mae_error = np.mean(np.abs(y_real - y_pred))
		normalization_value = np.sum(y_real)
		r2_score = 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)
		total_average_distance = np.mean([veh.distance for veh in self.fleet.vehicles])
		mean_idleness = np.mean(self.fleet.idleness_map)
		mean_weighted_idleness = np.mean(self.fleet.idleness_map * self.model.predict())
		coverage_percentage = np.sum(self.fleet.visited_map) / np.sum(self.navigation_map)
		true_reward = np.sum([self.true_reward[agent_id] for agent_id in self.fleet.vehicles_ids])
		
		return {'mse':                    mse_error,
		        'mae':                    mae_error,
		        'r2':                     r2_score,
		        'total_average_distance': total_average_distance,
		        'mean_idleness':          mean_idleness,
		        'mean_weighted_idleness': mean_weighted_idleness,
		        'coverage_percentage':    coverage_percentage,
		        'normalization_value':    normalization_value,
		        'true_reward':            true_reward,
		        }


if __name__ == "__main__":
	
	try:
		
		from PathPlanners.LawnMower import LawnMowerAgent
		from PathPlanners.NRRA import WanderingAgent
		import time
		
		scenario_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
		
		N = 4
		
		initial_positions = np.array([[42, 32],
		                              [50, 40],
		                              [43, 44],
		                              [35, 45]])
		
		env = DiscreteModelBasedPatrolling(n_agents=N,
		                                   navigation_map=scenario_map,
		                                   initial_positions=initial_positions,
		                                   model_based=True,
		                                   movement_length=2,
		                                   resolution=1,
		                                   influence_radius=2,
		                                   forgetting_factor=0.5,
		                                   max_distance=400,
		                                   benchmark='shekel',
		                                   dynamic=False,
		                                   reward_weights=[10, 10],
		                                   reward_type='weighted_idleness',
		                                   model='vaeUnet',
		                                   seed=50000,
		                                   int_observation=True)
		
		env.eval = True
		
		for m in range(10):
			
			t0 = time.time()
			env.reset()
			done = {i: False for i in range(N)}
			
			mse = []
			rewards_list = []
			
			agent = {i: WanderingAgent(world=scenario_map, number_of_actions=8, movement_length=4, seed=0) for i in
			         range(N)}
			
			while not all(done.values()):
				
				# actions = {i: np.random.randint(0,8) for i in done.keys() if not done[i]}
				actions = {i: agent[i].move(env.fleet.vehicles[i].position.astype(int)) for i in done.keys() if
				           not done[i]}
				observations, rewards, done, info = env.step(actions)
				
				for i in range(N):
					# If rewards dict does not contain the key, add it with 0 value #
					if i not in rewards.keys():
						rewards[i] = 0
				
				rewards_list.append([rewards[i] for i in range(N)])
				
				env.render()
				
				print("Rewards: ", rewards)
				print("Done: ", done)
				print("Info: ", info)
				
				plt.pause(0.2)
				mse.append(info['mse'])
			
			print("Time: ", time.time() - t0)
			
			plt.close()
			
			plt.figure()
			plt.plot(mse)
			plt.show()
	
	
	
	
	except KeyboardInterrupt:
		print("Interrupted")
		plt.close()
