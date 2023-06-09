import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from Environment.GroundTruths.AlgaeBloomGroundTruth import algae_bloom
from Environment.GroundTruths.ShekelGroundTruth import shekel
from gym import spaces

from Models.KNNmodel import KNNmodel, RKNNmodel
from Models.MiopicModel import MiopicModel
from Models.GaussianProcessModel import GaussianProcessModel
from Models.UnetModel import UnetDeepModel, benchmark_2_path

from sklearn.metrics import mean_squared_error, r2_score

class Vehicle:

	def __init__(self,
				 initial_positions: np.ndarray,
				 navigation_map: np.ndarray,
				 total_max_distance: float,
				 influence_radius: float,
				):
		
		# Copy the initial positions #
		self.initial_positions = np.atleast_2d(initial_positions).copy()
		self.navigation_map = navigation_map.copy()

		# Initialize positions
		self.position = None
		self.total_max_distance = total_max_distance
		self.distance = 0.0
		self.waypoints = []

		self.influence_radius = influence_radius

		self.influence_mask = np.zeros_like(self.navigation_map)

	def reset(self):

		# Get a random position from the initial  positions
		self.position = self.initial_positions[np.random.randint(0,len(self.initial_positions))]

		# Reset the distance 
		self.distance = 0.0

		# Reset the path
		self.waypoints = [self.position]

		self.influence_mask = self._influence_mask()

	def _influence_mask(self):
		""" Create a 0 matrix with the size of the navigation map and set to 1 a circle centered in the position of the vehicle of radius self.influence_radius"""

		influence_mask = np.zeros_like(self.navigation_map)

		# Set the influence mask to 1 in the circle centered in the position of the vehicle #

		# Compute the coordinates of the circle #
		x, y = np.meshgrid(np.arange(0, influence_mask.shape[1]), np.arange(0, influence_mask.shape[0]))
		x = x - self.position[1]
		y = y - self.position[0]

		# Compute the distance from the center #
		distance = np.sqrt(x**2 + y**2)

		# Set to 1 the points in the circle #
		influence_mask[distance <= self.influence_radius] = 1
		


		return influence_mask
	 
	def move(self, length, angle):
		# Take a step in given direction #

		direction = np.array([np.cos(angle), np.sin(angle)])

		next_target_position = length * direction + self.position

		# Check if the next_targer_position is visitable #

		collision = self.collision(next_target_position)

		if collision:

			# Obtain the nearest visitable point in the direction of the movement #

			last_truncated_position = self.position.copy()

			for i in np.arange(0, length+1):

				new_truncated_position = self.position + i * direction
				
				if self.collision(new_truncated_position):
					self.position = last_truncated_position.copy()
					break
				else:
					last_truncated_position = new_truncated_position.copy()

			self.waypoints.append(self.position)        
			self.influence_mask = self._influence_mask()

			return "COLLISION"

		else:
			
			# Compute distance #
			self.distance += np.linalg.norm(next_target_position - self.position)
			
			# Copy the next position #
			self.position = next_target_position.copy()
			self.waypoints.append(self.position)      
			self.influence_mask = self._influence_mask()  

			return "OK"
	
	def collision(self, position):
		# Check if the vehicle has collided #
		c_position = position.copy().astype(int)

		in_bound = c_position[0] >= 0 and c_position[0] < self.navigation_map.shape[0] and c_position[1] >= 0 and c_position[1] < self.navigation_map.shape[1]

		if not in_bound:
			return True
		else:
			return not self.navigation_map[c_position[0], c_position[1]].astype(bool)


class CoordinatedFleet:

	def __init__(self,
				 n_vehicles: int,
				 initial_positions: np.ndarray,
				 navigation_map: np.ndarray,
				 total_max_distance: float,
				 influence_radius: float,
				 max_num_of_steps: int,
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
		for vehicle in self.vehicles:
			vehicle.reset()

		self.idleness_map = np.ones_like(self.navigation_map)
		self.idleness_map_ = np.ones_like(self.navigation_map)

	def move(self, movements):

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
			self.vehicles[vehicle_id].move(movements[vehicle_id]['length'], movements[vehicle_id]['angle'])

		self.update_idleness_map()

	def get_vehicle_position_map(self, observer: int):
		""" Get the map with the observed position of the agent """
		
		position_map = np.zeros_like(self.navigation_map)
		position_map[self.vehicles[observer].position[0].astype(int), self.vehicles[observer].position[1].astype(int)] = 1

		return position_map
	
	def get_fleet_position_map(self, observers: np.ndarray):
		""" Get the map with the observed position of the agent """
		
		observers = np.atleast_1d(observers)
		position_map = np.zeros_like(self.navigation_map)

		for agent_id in self.vehicles_ids:
			
			if agent_id in observers:
				continue

			position_map[self.vehicles[agent_id].position[0].astype(int), self.vehicles[agent_id].position[1].astype(int)] = 1

		return position_map

	def get_positions(self):
		""" Return an arrey with the XY positions of the active agents """
		return np.array([self.vehicles[vehicle_id].position for vehicle_id in self.vehicles_ids])


	def render(self):


		if self.fig is None:

			self.fig, self.ax = plt.subplots(1,1)

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

	def __del__(self):
		
		if self.fig is not None:
			plt.close()

	def redundancy_mask(self):

		# Sum all the influence masks #
		return np.array([self.vehicles[i].influence_mask for i in self.vehicles_ids]).sum(axis=0) 
	
	def changes_idleness(self):

		# Compute the changes in idleness #
		net_change = np.abs(self.idleness_map - self.idleness_map_)

		# Compute the changes for every vehicle by vehicle mask multiplication #
		return {i: np.sum(self.vehicles[i].influence_mask * (net_change / np.sum(self.redundancy_mask()))) for i in self.vehicles_ids}

	def update_idleness_map(self):
		""" Update the idleness map """

		# Copy the previous idleness map #
		self.idleness_map_ = self.idleness_map.copy()

		# Update the idleness map #
		self.idleness_map += 1/self.max_num_of_steps  # Increment the idleness map everywhere

		self.idleness_map = np.clip(self.idleness_map, 0, 1)  # Clip the idleness map

		# Reset the idleness map in the vehicles influence area #
		for vehicle_id in self.vehicles_ids:
			self.idleness_map[np.where(self.vehicles[vehicle_id].influence_mask != 0)] = 0
	

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
				reward_type = 'weighted_importance',
				reward_weights = (1,1),
				benchmark: str = 'algae_bloom',
				model: str = 'miopic',
				dynamic: bool = True,
				seed: int = 0,
				random_gt: bool = True,
				):
		
		""" Copy attributes """

		np.random.seed(seed)	# Initialize the random seed

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

		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1))

		self.lambda_W = reward_weights[0]
		self.lambda_I = reward_weights[1]

		""" Create the fleet """
		self.fleet = CoordinatedFleet(n_vehicles=self.n_agents,
									  initial_positions=self.initial_positions,
									  navigation_map=self.navigation_map,
									  total_max_distance=self.max_distance,
									  influence_radius=self.influence_radius,
									  max_num_of_steps=(forgetting_factor*self.max_distance)//self.move_length,)
		
		# Compute the normalization value #
		example_vehicle = Vehicle(initial_positions=np.asarray([50,50]),
								navigation_map=np.ones((100,100)),
								total_max_distance=self.max_distance,
								influence_radius=self.influence_radius)
		example_vehicle.reset()

		influence_mask_0 = example_vehicle._influence_mask()
		example_vehicle.move(self.movement_length, 0)
		influence_mask_1 = example_vehicle._influence_mask()

		self.normalization_value = np.sum(np.clip(influence_mask_1 - influence_mask_0,0,1))

		""" Create the observation space """

		self.action_space = spaces.Discrete(8)

		self.observation_space = spaces.Box(low=0, high=1, shape=(5, self.navigation_map.shape[0], self.navigation_map.shape[1]), dtype=np.float32)

		""" Create the reward function """
		self.reward_type = reward_type

		self.model_str = model

		""" Create the model """
		if model == 'knn':
			self.model = KNNmodel(navigation_map=self.navigation_map, resolution=self.resolution, influence_radius=self.influence_radius, dt = 0.01)
		elif model == 'miopic':
			self.model = MiopicModel(navigation_map=self.navigation_map, resolution=self.resolution, influence_radius=self.influence_radius, dt = 0.7)
		elif model == 'rknn':
			self.model = RKNNmodel(navigation_map=self.navigation_map, resolution=self.resolution, influence_radius=self.influence_radius*2, dt = 0.01)
		elif model == 'deepUnet':
			self.model = UnetDeepModel(navigation_map=self.navigation_map, model_path = benchmark_2_path[benchmark], resolution=self.resolution, influence_radius=self.influence_radius, dt = 0.01)
		elif model == 'gp':
			self.model = GaussianProcessModel(navigation_map=self.navigation_map, resolution=self.resolution, influence_radius=self.influence_radius, dt = 0.01)
		else:
			raise ValueError('Unknown model')

			  					
		self.fig = None

		""" Create the benchmark """
		if benchmark == 'shekel':
			self.ground_truth = shekel(self.navigation_map, max_number_of_peaks=6, seed = self.seed, dt=0.05)
		elif benchmark == 'algae_bloom':
			self.ground_truth = algae_bloom(self.navigation_map, dt=0.5, seed=self.seed)
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

		self.steps = 0

		return self.get_observations()

	def action_to_movement(self, action: int):
		""" Convert the action to a movement order """

		return {'length': self.move_length, 'angle': action * 2.0 * np.pi / 8.0}
	
	def random_action(self):
		""" Return a random action """
		return self.action_to_movement(np.random.randint(8))

	def step(self, actions: dict):

		self.steps += 1

		movements_orders = {key: self.action_to_movement(action) for key, action in actions.items()}

		# Move the fleet #
		self.fleet.move(movements_orders)

		# Update the model #
		self.update_model()

		# Get the observations #
		observations = self.get_observations()

		# Get the rewards #
		rewards = self.get_rewards()

		# Get the done flag #
		done = self.get_done()

		# Get the info #
		self.update_info()

		if self.dynamic:
			self.ground_truth.step()

		return observations, rewards, done, self.info
	
	def update_model(self):
		""" Update the model """

		# Obtain all the new positions of the agents #
		positions = self.fleet.get_positions()

		# Obtain the values of the ground truth in the new positions #
		values = self.ground_truth.read(positions)

		self.previous_model = self.model.predict().copy()

		# Update the model #
		if self.dynamic:
			self.model.update(positions, values, np.ones_like(values)*self.steps)
		elif self.model_str == 'deepUnet':
			self.model.update(positions, values, self.fleet.idleness_map)
		else:
			self.model.update(positions, values)

		

	
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
			
			observations[vehicle_id] = np.concatenate((
				self.navigation_map[np.newaxis],
				(0.5*self.fleet.get_vehicle_position_map(observer=vehicle_id) + 0.5*self.fleet.vehicles[vehicle_id].influence_mask)[np.newaxis],
				self.fleet.get_fleet_position_map(observers=vehicle_id)[np.newaxis],
				self.fleet.idleness_map[np.newaxis],
				self.model.predict()[np.newaxis],
				), axis=0)

		self.observations = observations

		return self.observations
	
	def get_rewards(self):
		""" The reward is selected dependign on the reward type """

		reward = {}
		self.info['W'] = 0.0
		self.info['I'] = 0.0

		if self.reward_type == 'local_changes':

			
			# Compute the reward as the local changes for the agent. #

			W = self.fleet.changes_idleness() # Compute the idleness changes #
		

			for agent_id in self.fleet.vehicles_ids:
				
				# The reward is the sum of the local changes in the agent's influence area + W #

				if self.dynamic:
					difference_map = np.abs(self.model.predict() - self.previous_model)
					information_gain = np.sum(self.fleet.vehicles[agent_id].influence_mask * (difference_map / np.sum(self.fleet.redundancy_mask())))
				else:
					information_gain = np.sum(self.fleet.vehicles[agent_id].influence_mask * (self.model.predict() / np.sum(self.fleet.redundancy_mask())))

				reward[agent_id] = (self.lambda_W * W[agent_id] + self.lambda_I * information_gain)
				self.info['W'] += W[agent_id]
				self.info['I'] += information_gain
		else:
			# TODO: Implement other reward types #
			raise NotImplementedError
			

		return reward
	
	def get_done(self):
		""" End the episode when the distance is greater than the max distance """

		done = {agent_id : self.fleet.vehicles[agent_id].distance > self.max_distance for agent_id in self.fleet.vehicles_ids}

		return done
	
	def update_info(self):


		y_pred = self.model.predict()[self.visitable_positions[:,0], self.visitable_positions[:,1]].flatten()
		y_true = self.ground_truth.read()[self.visitable_positions[:,0], self.visitable_positions[:,1]].flatten()

		self.info['mse'] = mean_squared_error(y_pred, y_true, squared=True)
		self.info['rmse'] = mean_squared_error(y_pred, y_true, squared=False)
		self.info['weighted_rmse'] = mean_squared_error(y_pred, y_true, squared=False, sample_weight=np.clip(y_true, 0.1, 1.0))
		self.info['R2'] = r2_score(y_pred, y_true)

		return self.info
	
	def render(self):

		if self.fig is None:

			self.fig, self.axs = plt.subplots(3,2)
			self.axs = self.axs.flatten()

			# Plot the navigation map #
			#self.d0 = self.axs[0].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][0], cmap='gray', vmin=0, vmax=1)
			self.d0 = self.axs[0].imshow(self.ground_truth.read(), cmap='gray', vmin=0, vmax=1)
			self.axs[0].set_title('Navigation map')
			self.d1 = self.axs[1].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][1], cmap='gray', vmin=0, vmax=1)
			self.axs[1].set_title('Agent Position')
			self.d2 = self.axs[2].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][2], cmap='gray', vmin=0, vmax=1)
			self.axs[2].set_title('Fleet Position')
			self.d3 = self.axs[3].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][3], cmap='gray', vmin=0, vmax=1)
			self.axs[3].set_title('Model')
			self.d4 = self.axs[4].imshow(self.observations[list(self.fleet.vehicles_ids)[0]][4], cmap='gray', vmin=0, vmax=1)
			self.axs[4].set_title('idleness')

		else:

			#self.d0.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][0])
			self.d0.set_data(self.ground_truth.read())
			self.d1.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][1])
			self.d2.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][2])
			self.d3.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][3])
			self.d4.set_data(self.observations[list(self.fleet.vehicles_ids)[0]][4])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		plt.pause(0.01)





if __name__ == "__main__":

	from PathPlanners.LawnMower import LawnMowerAgent
	from PathPlanners.NRRA import WanderingAgent

	scenario_map = np.genfromtxt('Environment\Maps\map.txt', delimiter=' ')

	N = 4

	initial_positions = np.array([[42,32],
								[50,40],
								[43,44],
								[35,45]])

	env = DiscreteModelBasedPatrolling(n_agents=N,
								navigation_map=scenario_map,
								initial_positions=initial_positions,
								model_based=True,
								movement_length=3,
								resolution=1,
								influence_radius=2,
								forgetting_factor=2,
								max_distance=100,
								benchmark='shekel',
								dynamic=False,
								reward_weights=[10, 10],
								reward_type='local_changes',
								model='miopic',
								seed=50000,
								)

	for m in range(10):
		
		env.reset()
		env.reset()
		done = {i: False for i in range(N)}

		mse = []
		rewards_list = []
		#agent = {i: LawnMowerAgent( world=scenario_map, number_of_actions=8, movement_length= 3, forward_direction=0, seed=0) for i in range(N)}
		agent = {i: WanderingAgent( world=scenario_map, number_of_actions=8, movement_length= 3, seed=0) for i in range(N)}
		
		while not all(done.values()):

			#actions = {i: np.random.randint(0,8) for i in done.keys() if not done[i]}
			actions = {i: agent[i].move(env.fleet.vehicles[i].position.astype(int)) for i in done.keys() if not done[i]}
			observations, rewards, done, info = env.step(actions)

			print(env.normalization_value)

			for i in range(N):
				# If rewards dict does not contain the key, add it with 0 value #
				if i not in rewards.keys():
					rewards[i] = 0

			rewards_list.append([rewards[i] for i in range(N)])

			env.render()
			print("Rewards: ", rewards)
			print("Done: ", done)
			print("Info: ", info)

			mse.append(np.mean(np.sqrt(info['mse'])))

		
		plt.close()

		plt.plot(mse)

		plt.show()

		plt.close()

		plt.plot(np.cumsum(np.asarray(rewards_list), axis =0))

		plt.show()
		