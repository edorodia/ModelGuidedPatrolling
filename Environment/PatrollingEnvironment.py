import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Environment.GroundTruths.AlgaeBloomGroundTruth import algae_bloom
from Environment.GroundTruths.ShekelGroundTruth import shekel
from gym import spaces
from Models.KNNmodel import KNNmodel, RKNNmodel
from Models.MiopicModel import MiopicModel
from HetModels.HetMiopicModel import HetMiopicModel
from Models.GaussianProcessModel import GaussianProcessModel
from Models.PolinomialRegressor import PolinomialRegressor
from Models.SVRegressor import SVRegressor
from Models.UnetModel import UnetDeepModel, VAEUnetDeepModel, benchmark_2_vae_path
from HetModels.HetUnetModel import VAEUnetDeepHetModel, benchmark_2_vae_path_het
from sklearn.metrics import mean_squared_error, r2_score
from PathPlanners.dijkstra import Dijkstra

from Environment.exploration_policies import preComputedExplorationPolicy

from HetModels.NoiseModels.NoNoise import NoNoise
from HetModels.NoiseModels.MeanNoise import MeanNoise
from HetModels.NoiseModels.FishEyeNoiseApproximator import FishEyeNoiseApproximator

class Drone:
	#influence_side: defines the number of cells in the navigation map covered by a single picture of the drone (the equivalent of the influence_radius of an ASV)
	#camera_fov_angle: FOV angle of the camera mounted on the drone
	#drone_height: height of the drone during operational phase
	#blur_data: boolean flag to set whether to use average on all the cells or to set differentiated values for every cell
	def __init__(self,
	             initial_positions: np.ndarray,
	             navigation_map: np.ndarray,
	             total_max_distance: float,
				 influence_side: float,
	             camera_fov_angle: float,
				 drone_height: float,
				 blur_data: bool):
		
		# Copy the initial positions #
		self.initial_positions = np.atleast_2d(initial_positions).copy()
		self.navigation_map = navigation_map.copy()
		
		# Initialize positions
		self.position = None
		self.total_max_distance = total_max_distance
		self.distance = 0.0
		self.waypoints = []
		self.last_waypoints = []
		self.steps = 0

		self.influence_side = influence_side
		
		
		self.camera_fov_angle = camera_fov_angle
		self.drone_height = drone_height
		self.blur_data = blur_data
		
		#Generates a new structures equal to the one given in input but filled only with 0
		self.influence_mask = np.zeros_like(self.navigation_map)

	def reset(self):
		# Get a random position from the initial positions
		self.position = self.initial_positions[np.random.randint(0, len(self.initial_positions))]

		# Reset the distance
		self.distance = 0.0
		
		# Reset the path
		self.waypoints = [self.position]
		self.last_waypoints = [self.position]
		
		self.influence_mask = self._influence_mask()
		
		self.steps = 0
	
	#draws influence mask of the drones using the influence_side
	def _influence_mask(self):
		""" Create a 0 matrix with the size of the navigation map and set to 1 a square centered in the position of
		the drone of side size influence_side """
		
		influence_mask = np.zeros_like(self.navigation_map)
		
		offset = self.influence_side/2

		x_start 	= int(np.ceil(self.position[1] - offset))
		if x_start <= 0:
			x_start = 0
		y_start 	= int(np.ceil(self.position[0] - offset))
		if y_start <= 0:
			y_start = 0

		x_end 		= int(np.floor(self.position[1] + offset))
		y_end 		= int(np.floor(self.position[0] + offset))

		influence_mask[y_start : y_end + 1, x_start : x_end + 1] = 1

		return influence_mask
	
	#calculataes the side size (meters) of the square of surface covered by a single picture take by the drone using the law of sines (aka sine rule)
	def square_size(self):
		return np.floor(2*((self.drone_height/np.sin(np.radians(180-90-(self.camera_fov_angle/2))))*np.sin(np.radians(self.camera_fov_angle/2))))

	#method that allows to move the drone to a cell on the lake given it's position in coordinates
	def move(self, x:int, y:int):			
			self.steps += 1
			
			next_target_position = np.array([y,x])
   
			collision = self.collision(next_target_position)
			
			if collision:
				#puts in the last_waypoints the last valid waypoint reached
				return "COLLISION"
			else:
				self.last_waypoints = []
				self.distance += np.linalg.norm(next_target_position - self.position)
				self.position = next_target_position
				self.waypoints.append(next_target_position)
				self.last_waypoints.append(next_target_position)
				self.influence_mask = self._influence_mask()
				self.last_waypoints = np.asarray(self.last_waypoints)
			return "OK"

	def collision(self, position):
			# Check if the drone can or cannot go to position #
			c_position = position.copy().astype(int)
			
			in_bound = (0 <= int(c_position[0]) < self.navigation_map.shape[0] and 0 <= int(c_position[1]) <
						self.navigation_map.shape[1])
			
			if not in_bound:
				return True
			else:
				return not self.navigation_map[c_position[0], c_position[1]].astype(bool)


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
		
		#Generates a new structures equal to the one given in input but filled only with 0
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
		
		# Compute the distances from the center to the other points#
		distance = np.sqrt(x ** 2 + y ** 2)
		
		# Set to 1 the points in the circle #
		influence_mask[distance <= self.influence_radius] = 1
		
		return influence_mask
	
	#requires the angle expressed in radians
	def move(self, length, angle, action_type='discrete'):
		# Take a step in given direction #
		
		#increases the steps counter
		self.steps += 1
		

		#calculates the discrete direction x and y of the movement, the two data calculated will be summed to the actual position already given in x and y terms
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
	             forgetting_factor: float,
				 influence_asv_visited_map: bool = False,
				 check_max_distances: bool = True
	             ):
		
		self.navigation_map = navigation_map
		self.initial_positions = initial_positions
		self.total_max_distance = total_max_distance
		self.n_vehicles = n_vehicles
		self.forgetting_factor = forgetting_factor

		self.check_max_distances = check_max_distances

		self.influence_radius = influence_radius
		
		self.influence_asv_visited_map = influence_asv_visited_map

		self.idleness_map = np.ones_like(self.navigation_map)
		self.idleness_map_ = np.ones_like(self.navigation_map)
		self.changes_in_idleness = np.zeros_like(self.navigation_map)

		self.visitable_positions = np.array(np.where(self.navigation_map == 1)).T
		
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
			"""
			first position is not signed in the visited_map because there is no corresponding importance read
			self.visited_map[vehicle.position[0].astype(int), vehicle.position[1].astype(int)] = 1
			"""
			
		self.idleness_map = np.ones_like(self.navigation_map)
		self.idleness_map_ = np.ones_like(self.navigation_map)
	
	def move(self, movements, action_type='discrete'):
		
		# Set the flags for inactive agents #
		remove_ids = []
		for vehicle_id in self.vehicles_ids:
			if self.check_max_distances:
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
			if self.influence_asv_visited_map and self.influence_radius != 0:
				#tracks the influence radius if there is an influence radius and the flag is set
				positions_veh = self.vehicles[vehicle_id].last_waypoints.astype(int)
				for i in range(len(positions_veh)):
					# Compute all the distances between visitable positions and the positions of the i-th robot
					distances = np.linalg.norm(self.visitable_positions - positions_veh[i].astype(int), axis=1).astype(float)
					# Get the positions that are closer than influence_radius
					positions = self.visitable_positions[distances <= self.influence_radius]

					# Extract rows and columns from positions
					rows = positions[:, 0]
					cols = positions[:, 1]

					# Check if the indices are within bounds
					row_mask = (rows >= 0) & (rows < self.visited_air_map.shape[0])
					col_mask = (cols >= 0) & (cols < self.visited_air_map.shape[1])

					# Combined mask for both row and column bounds
					valid_mask = row_mask & col_mask

					# Extract valid indices
					valid_rows = rows[valid_mask]
					valid_cols = cols[valid_mask]

					# Set the positions to y
					# Sets the positions under the influence radius to the values discovered
					self.visited_map[positions[:,0], positions[:,1]] = 1
				
				#this is what is done if there is no influence radius every position in the model map gets it corresponding value withouth considering the radius
				self.visited_map[positions_veh[:,0].astype(int), positions_veh[:,1].astype(int)] = 1
			else:
				#tracks only the real positions
				"""it does it by taking the x and y data about the last waypoints explored for every vehicle and setting
				that position to 1 in the visited_map"""
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
		clipped_redundancy = np.clip(self.redundancy_mask(), 1 , np.inf)
		return {i: (self.vehicles[i].influence_mask * (net_change / clipped_redundancy)) for i in
		        self.vehicles_ids}
	
	def update_idleness_map(self):
		""" Update the idleness map """
		
		# Copy the previous idleness map #
		self.idleness_map_ = self.idleness_map.copy()
		
		# Update the idleness map #
		self.idleness_map += self.forgetting_factor # Increment the idleness map everywhere
		
		self.idleness_map = np.clip(self.idleness_map, 0, 1)  # Clip the idleness map
		
		# Reset the idleness map in the vehicles influence area #
		for vehicle_id in self.vehicles_ids:
			self.idleness_map[np.where(self.vehicles[vehicle_id].influence_mask != 0)] = 0
		
		self.idleness_map = self.idleness_map * self.navigation_map
	
	def get_last_waypoints(self):
		""" Return the last waypoints of the vehicles """
		# till here it works
		return np.vstack([vehicle.last_waypoints for vehicle in self.vehicles])


class CoordinatedHetFleet(CoordinatedFleet):
	def __init__(self,
	             n_vehicles: int,							
	             initial_surface_positions: np.ndarray,		
	             navigation_map: np.ndarray,				
	             total_max_surface_distance: float,			
	             influence_radius: float,					
	             forgetting_factor: float,	
				 air_forgetting_factor: float,	
				 initial_air_positions: np.ndarray,			
				 total_max_air_distance: float,				
				 influence_side: float,						
	             camera_fov_angle: float,					
				 drone_height: float,						
				 blur_data: bool,	
				 drone_idleness_influence: float,		
				 drone_direct_idleness_influece: bool,					
				 n_drones: int,
				 update_only_with_ASV: bool,
				 influence_drone_visited_map: bool,
				 influence_asv_visited_map: bool,
				 check_max_distances: bool = True):						

		super().__init__(n_vehicles, 
				       initial_surface_positions,
					   navigation_map,
					   total_max_surface_distance,
					   influence_radius,
					   forgetting_factor,
					   influence_asv_visited_map = influence_asv_visited_map,
					   check_max_distances = check_max_distances)

		self.total_max_air_distance = total_max_air_distance
		self.initial_air_positions = initial_air_positions
		self.air_forgetting_factor = air_forgetting_factor
		self.drone_direct_idleness_influece = drone_direct_idleness_influece
		self.drone_idleness_influence = drone_idleness_influence
		self.influence_side = influence_side
		self.n_drones = n_drones

		self.check_max_distances = check_max_distances

		self.update_only_with_ASV = update_only_with_ASV
		self.influence_drone_visited_map = influence_drone_visited_map
		self.air_idleness_edited = False 

		self.idleness_air_map = np.ones_like(self.navigation_map)
		self.idleness_air_map_ = np.ones_like(self.navigation_map)
		self.changes_in_air_idleness = np.zeros_like(self.navigation_map)

		# Create the aerial fleet #
		self.drones = [Drone(initial_positions=initial_air_positions[i],
		                         navigation_map=navigation_map,
		                         total_max_distance=total_max_air_distance,
		                         influence_side=influence_side,
								 camera_fov_angle=camera_fov_angle,
								 drone_height=drone_height,
								 blur_data=blur_data) for i in range(n_drones)]
		
		self.drones_ids = set(range(n_drones))

	def reset(self):
		super().reset()

		self.drones_ids = set(range(self.n_drones))
		self.visited_air_map = np.zeros_like(self.navigation_map)
		
		for drone in self.drones:
			drone.reset()
		"""
			this part is commented out because otherwise at reset, the visited air map would have a drone square without making the importance read in the ground truth causing a difference between two metrics importance and visited map 
			if self.influence_drone_visited_map :
				offset = self.influence_side/2

				position = drone.position.astype(int)
			
				column_start = int(np.ceil(position[1] - offset))
				row_start 	 = int(np.ceil(position[0] - offset))

				column_end 	 = int(np.floor(position[1] + offset))
				row_end 	 = int(np.floor(position[0] + offset))

				column_grid  = np.arange(column_start, column_end + 1)
				row_grid     = np.arange(row_start, row_end + 1)

				grid1, grid2 = np.meshgrid(row_grid, column_grid)

				positions = np.column_stack((grid1.ravel(), grid2.ravel()))

				# Extract rows and columns from positions
				rows = positions[:, 0]
				cols = positions[:, 1]

				# Check if the indices are within bounds
				row_mask = (rows >= 0) & (rows < self.visited_air_map.shape[0])
				col_mask = (cols >= 0) & (cols < self.visited_air_map.shape[1])

				# Combined mask for both row and column bounds
				valid_mask = row_mask & col_mask

				# Initialize result array with zeros
				result = np.zeros(len(positions))

				# Extract valid indices
				valid_rows = rows[valid_mask]
				valid_cols = cols[valid_mask]

				self.visited_air_map[valid_rows.astype(int), valid_cols.astype(int)] = 1
			else :
				self.visited_air_map[drone.position[0].astype(int), drone.position[1].astype(int)] = 1
		"""
		self.idleness_air_map = np.ones_like(self.navigation_map)
		self.idleness_air_map_ = np.ones_like(self.navigation_map)

	def move_ASVs(self, movements, action_type='discrete'):
		super().move(movements, action_type)
	
	# to_positions : is a dictionary that has only the drone:position values of the drones that have to move
	def move_Drones(self, to_positions: dict):

		# Set the flags for inactive drones that have to move #
		remove_ids = []
		for drone_id in to_positions.keys():
			if self.check_max_distances:
				if self.drones[drone_id].distance > self.total_max_air_distance:
					remove_ids.append(drone_id)
		
		# Remove the inactive drones from the drones_ids list #
		for drone_id in remove_ids:
			self.drones_ids.remove(drone_id)
		
		# Move the drones #
		for drone_id in to_positions.keys():
			if drone_id in self.drones_ids:
				# Move the drone_id drone to the position in its index #
				self.drones[drone_id].move(to_positions[drone_id][1], to_positions[drone_id][0])
				
		for drone_id in to_positions.keys():
			# Update the visited map #
			"""it does it by taking the x and y data about the last waypoints explored for every vehicle and setting
			that position to 1 in the visited_map"""
			
			if self.influence_drone_visited_map :

				positions = self.drones[drone_id].last_waypoints.astype(int)
				
				for position in positions:

					offset = self.influence_side/2
				
					column_start = int(np.ceil(position[1] - offset))
					row_start 	 = int(np.ceil(position[0] - offset))

					column_end 	 = int(np.floor(position[1] + offset))
					row_end 	 = int(np.floor(position[0] + offset))

					column_grid  = np.arange(column_start, column_end + 1)
					row_grid     = np.arange(row_start, row_end + 1)

					grid1, grid2 = np.meshgrid(row_grid, column_grid)

					final_pos = np.column_stack((grid1.ravel(), grid2.ravel()))

					# Extract rows and columns from positions
					rows = final_pos[:, 0]
					cols = final_pos[:, 1]

					# Check if the indices are within bounds
					row_mask = (rows >= 0) & (rows < self.visited_air_map.shape[0])
					col_mask = (cols >= 0) & (cols < self.visited_air_map.shape[1])

					# Combined mask for both row and column bounds
					valid_mask = row_mask & col_mask

					# Extract valid indices
					valid_rows = rows[valid_mask]
					valid_cols = cols[valid_mask]

					self.visited_air_map[valid_rows.astype(int), valid_cols.astype(int)] = 1
			
			else:

				self.visited_air_map[
					self.drones[drone_id].last_waypoints[:, 0].astype(int),
					self.drones[drone_id].last_waypoints[:, 1].astype(int)] = 1
		
		self.update_idleness_map_drone()

	""" the update idleness function when ASVs are moved """
	def update_idleness_map(self):
		""" ASV's idleness has to be reset in the ASV influence area """
		# this is done in the super's .move method #
		super().update_idleness_map()
		
		""" Drone's idleness has to be reset in the ASV influence area """
		# Copy the previous idleness air map #
		self.idleness_air_map_ = self.idleness_air_map.copy()
		
		# Checks if the air forgetting factor was already added by the other method #
		if not self.air_idleness_edited:

			# Update the idleness air map #
			self.idleness_air_map += self.air_forgetting_factor # Increment the idleness air map everywhere
			
			self.idleness_air_map = np.clip(self.idleness_air_map, 0, 1)  # Clip the idleness map
			self.air_idleness_edited = True

		# Reset the idleness air map in the vehicles influence area #
		for vehicle_id in self.vehicles_ids:
			self.idleness_air_map[np.where(self.vehicles[vehicle_id].influence_mask != 0)] = 0
		
		self.idleness_air_map = self.idleness_air_map * self.navigation_map

	""" the update idleness function when drone is moved """
	def update_idleness_map_drone(self):
		""" ASV's idleness has to be reduced in the influence area of the 
			drone if the flag drone_direct_idleness_influence is set to true """
		if self.drone_direct_idleness_influece :
			# if the drone has direct influence on ASV idleness then do this #
			# Copy the previous idleness air map #
			self.idleness_map_ = self.idleness_map.copy()
			
			# Reset the idleness air map in the vehicles influence area #
			for drone_id in self.drones_ids:
				self.idleness_map[np.where(self.drones[drone_id].influence_mask != 0)] -= self.drone_idleness_influence
			
			self.idleness_map = self.idleness_map * self.navigation_map


		""" Drone's idleness has to be reset to 0 in the Drone influence area """
		# Copy the previous idleness air map #
		self.idleness_air_map_ = self.idleness_air_map.copy()

		# Checks if the air forgetting factor was already added by the other method #
		if not self.air_idleness_edited and not self.update_only_with_ASV:
			
			# Update the idleness air map #
			self.idleness_air_map += self.air_forgetting_factor # Increment the idleness air map everywhere
			
			self.idleness_air_map = np.clip(self.idleness_air_map, 0, 1)  # Clip the idleness map
			self.air_idleness_edited = True
		
		# Reset the idleness air map in the vehicles influence area #
		for drone_id in self.drones_ids:
			self.idleness_air_map[np.where(self.drones[drone_id].influence_mask != 0)] = 0
		
		self.idleness_air_map = self.idleness_air_map * self.navigation_map

	def end_step_movements(self):
		self.air_idleness_edited = False
	
	def get_ASV_position_map(self, observer: int):
		return super().get_vehicle_position_map(observer)
	
	def get_drone_position_map(self, observer: int):
		""" Get the map with the observed position of the drone """
		
		position_map = np.zeros_like(self.navigation_map)
		position_map[
			self.drones[observer].position[0].astype(int), self.drones[observer].position[1].astype(int)] = 1
		
		return position_map

	def get_ASV_trajectory_map(self, observer: int):
		return super().get_vehicle_trajectory_map(observer)
	
	def get_drone_trajectory_map(self, observer: int):
		position_map = np.zeros_like(self.navigation_map)
		
		waypoints = np.asarray(self.drones[observer].waypoints)
		
		position_map[waypoints[-10:, 0].astype(int), waypoints[-10:, 1].astype(int)] = np.linspace(0, 1,
		                                                                                           len(waypoints[
		                                                                                               -10:]),
		                                                                                           endpoint=True)
		
		return position_map
		
	def get_ASV_fleet_position_map(self, observers: int):
		super().get_fleet_position_map(observers)
	
	def get_drone_fleet_position_map(self, observers: int):
		""" Get the map with the observed position of the drones """
		
		observers = np.atleast_1d(observers)
		position_map = np.zeros_like(self.navigation_map)
		
		for drone_id in self.drones_ids:
			
			if drone_id in observers:
				continue
			
			position_map[
				self.drones[drone_id].position[0].astype(int), self.drones[drone_id].position[1].astype(int)] = 1
		
		return position_map

	def get_ASV_positions(self):
		return super().get_positions()
	
	def get_drone_positions(self):
		""" Return an array with the XY positions of the active drones """
		return np.array([self.drones[drone_id].position for drone_id in self.drones_ids])

	def redundancy_ASV_mask(self):
		# Sum all the influence masks #
		# to avoid further problems the zones with a 0 are replaced with a 1 this doesn't change anything since influence masks will mask the values to not care about
		
		#redundancy_mask = np.array([self.vehicles[i].influence_mask for i in self.vehicles_ids]).sum(axis=0)
		#redundancy_mask[redundancy_mask == 0] = 1
		#return redundancy_mask

		return np.array([self.vehicles[i].influence_mask for i in self.vehicles_ids]).sum(axis=0)

	def redundancy_drone_mask(self):
		# Sum all the influence masks #
		# Generates an array of influence masks and sums all of them together (the function sums all the rows with same index together)
		# to avoid further problems the zones with a 0 are replaced with a 1 this doesn't change anything since influence masks will mask the values to not care about
		
		#redundancy_mask = np.array([self.drones[i].influence_mask for i in self.drones_ids]).sum(axis=0)
		#redundancy_mask[redundancy_mask == 0] = 1
		#return redundancy_mask

		return np.array([self.vehicles[i].influence_mask for i in self.vehicles_ids]).sum(axis=0)

	def changes_ASV_idleness(self):
		# Compute the changes in idleness #
		net_change = np.abs(self.idleness_map - self.idleness_map_)
		
		# Compute the changes for every vehicle by vehicle mask multiplication #
		#return {i: np.sum(self.vehicles[i].influence_mask * (net_change / self.redundancy_mask())) for i in
		#        self.vehicles_ids}
		clipped_redundancy = np.clip(self.redundancy_ASV_mask(), 1 , np.inf)
		return {i: (self.vehicles[i].influence_mask * (net_change / clipped_redundancy)) for i in
		        self.vehicles_ids}
	
	def changes_air_idleness(self):
		# Compute the changes in air idleness #
		net_air_change = np.abs(self.idleness_air_map - self.idleness_air_map_)
		
		# Compute the changes for every vehicle by vehicle mask multiplication #
		#return {i: np.sum(self.drones[i].influence_mask * (net_air_change / self.redundancy_drone_mask())) for i in
		#        self.drones_ids}
		clipped_redundancy = np.clip(self.redundancy_drone_mask(), 1 , np.inf)
		return {i: (self.drones[i].influence_mask * (net_air_change / clipped_redundancy)) for i in
		         self.drones_ids}

	def get_last_ASV_waypoints(self):
		return super().get_last_waypoints()
	
	def get_last_drone_waypoints(self):
		waypoints_dict = {}
		for drone_id in self.drones_ids:
			waypoints_dict[drone_id]=np.vstack([self.drones[drone_id].last_waypoints])
		""" Return the last waypoints of the drones """
		return waypoints_dict

	def render(self):
		
		if self.fig is None:
			
			#self.fig -> the entire window with all the figures
			#self.ax -> single plotting area, the subplot
			self.fig, self.ax = plt.subplots(1, 1)
			
			self.ax_pos = []
			self.ax_pos_drone = []
			
			self.ax.imshow(self.navigation_map, cmap='gray', vmin=0, vmax=1)
			
			# Plot all vehicles' positions #
			for vehicle in self.vehicles:
				self.ax_pos.append(self.ax.plot(vehicle.position[1], vehicle.position[0], 'ro')[0])
			
			for drone in self.drones:
				self.ax_pos_drone.append(self.ax.plot(drone.position[1], drone.position[0], 'bo')[0])
			
			plt.show(block=False)
		else:
			
			# Update the positions #
			for vehicle_id in self.vehicles_ids:
				self.ax_pos[vehicle_id].set_xdata([self.vehicles[vehicle_id].position[1]])
				self.ax_pos[vehicle_id].set_ydata([self.vehicles[vehicle_id].position[0]])
			
			for drone_id in self.drones_ids:
				self.ax_pos_drone[drone_id].set_xdata([self.drones[drone_id].position[1]])
				self.ax_pos_drone[drone_id].set_ydata([self.drones[drone_id].position[0]])
		
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		plt.pause(0.1)


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
	             forgetting_factor: float = 0.01,
	             reward_type='weighted_importance',
	             reward_weights=(1, 1),
	             benchmark: str = 'algae_bloom',
	             model: str = 'miopic',
	             dynamic: bool = True,
	             seed: int = 0,
	             random_gt: bool = True,
	             int_observation: bool = False,
	             min_information_importance: float = 1.0,
	             previous_exploration=False,
	             pre_exploration_policy=None,
	             pre_exploration_steps=0,
				 check_max_distances: bool = True
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
		self.check_max_distances = check_max_distances
		
		self.true_reward = {}
		
		self.int_observation = int_observation
		
		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1))
		
		self.lambda_W = reward_weights[0]
		self.lambda_I = reward_weights[1]
		
		self.max_num_steps = (forgetting_factor * self.max_distance) // self.move_length
		
		self.max_agent_steps = self.max_distance // self.move_length + 1
		
		self.forgetting_factor = forgetting_factor
		
		""" Pre exploration Parameters """
		self.previous_exploration = previous_exploration
		self.pre_exploration_policy = pre_exploration_policy
		self.pre_exploration_steps = pre_exploration_steps
		
		if self.previous_exploration:
			self.initial_positions = self.pre_exploration_policy.initial_positions
		
		""" Create the fleet """
		self.fleet = CoordinatedFleet(n_vehicles=self.n_agents,
		                              initial_positions=self.initial_positions,
		                              navigation_map=self.navigation_map,
		                              total_max_distance=self.max_distance,
		                              influence_radius=self.influence_radius,
		                              forgetting_factor=self.forgetting_factor,
									  check_max_distances = self.check_max_distances)
		
		""" Create the observation space """
		
		#creates a set of 8 integer values from 0 to 7
		self.action_space = spaces.Discrete(8)
		
		#defines an observation_space made by a 2 dimensions matrix with 4 channels per cell
		"""
		Spaces are crucially used in Gym to define the format of valid actions and observations
		"""
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
		elif model == 'none':
			self.model = MiopicModel(navigation_map=self.navigation_map,
			                         resolution=self.resolution,
			                         influence_radius=0,
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
		
		if self.previous_exploration:
			""" If a pre-exploration policy is given, execute it """
			self.pre_exploration_policy.reset()
			for steps in range(self.pre_exploration_steps):
				actions = self.pre_exploration_policy.suggest_action()
				self.step(actions, action_type='next_position')
		
		return self.get_observations()
	
	def action_to_movement(self, action: int):
		""" Convert the action to a movement order """
		
		return {'length': self.move_length, 'angle': action * 2.0 * np.pi / 8.0}
	
	def random_action(self):
		""" Return a random action """
		return self.action_to_movement(np.random.randint(8))
	
	def step(self, actions: dict, action_type='discrete'):
		#increase the step counter by 1 unit
		self.steps += 1
		
		#converts action integer number to movement (angle and length)
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
		
		if self.eval:
			self.info = self.get_info()

		#if the dynamic flag is set calls for a step in the ground truth
		if self.dynamic:
			self.ground_truth.step()
		
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
			#print("Posizioni passate -> " + str(sample_positions))
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
		
		#for every vehicle
		for vehicle_id in self.fleet.vehicles_ids:
			
			if self.int_observation:
				
				#saves the observation for that vehicle in the i-th position
				#convert the values to values in the [0,255] range
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
		""" The reward is selected depending on the reward type """
		
		reward = {}
		
		self.true_reward = {}
		
		if self.reward_type == 'weighted_idleness':
			
			# Compute the reward as the local changes for the agent. #
			
			#it is a single value returned
			W = self.fleet.changes_idleness()  # Compute the idleness changes #
			
			for agent_id in self.fleet.vehicles_ids:
				# The reward is the sum of the local changes in the agent's influence area + W #
				
				#information gain calculated with only the predictive model
				information_gain = self.fleet.vehicles[agent_id].influence_mask * (
						self.model.predict() / self.fleet.redundancy_mask())
				
				#information gain calculated with the ground truth
				true_information_gain = self.fleet.vehicles[agent_id].influence_mask * (
						self.ground_truth.read() / self.fleet.redundancy_mask())
				
				#every agent has its own reward calculated and based upon its influence in the total environment
				#W is the change outside the actual influence of the agent, and this change is a difference so the higher it is the better
				#the information gain is based on the value of the zone that the agent is measuring, the quantity of algae that is present, so that has to be higher too
				#we need to maximize this reward
				reward[agent_id] = W[agent_id] * (information_gain + self.min_information_importance) * 100.0
				#the true reward is also calculated, the only difference is that true_information_gain is used instead of the previous one
				self.true_reward[agent_id] = W[agent_id] * (
						true_information_gain + self.min_information_importance) * 100.0
		
		else:
			raise NotImplementedError('Unknown reward type')
		#print(reward)
		return reward
	
	def get_done(self):
		""" End the episode when the distance is greater than the max distance """
		done = {agent_id: False for agent_id in self.fleet.vehicles_ids}

		if self.check_max_distances:
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
		
		#questo è l'errore medio tra una cella reale e la sua controparte rilevata dagli agenti
		mse_error = np.mean((y_real - y_pred) ** 2)
		mae_error = np.mean(np.abs(y_real - y_pred))
		normalization_value = np.sum(y_real)
		r2_score = 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)
		total_average_distance = np.mean([veh.distance for veh in self.fleet.vehicles])
		mean_idleness = np.sum(self.fleet.idleness_map)/np.sum(self.navigation_map)
		mean_weighted_idleness = np.sum(self.fleet.idleness_map * self.ground_truth.read())/np.sum(self.navigation_map)
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


class DiscreteModelBasedHetPatrolling(DiscreteModelBasedPatrolling):

	def __init__(self,
				 initial_air_positions: np.ndarray,					#
				 max_air_distance: float,							#
				 influence_side: float,								#
				 forgetting_air_factor: float,						#		
				 drone_idleness_influence : float,					#
	             n_agents: int,
	             navigation_map: np.ndarray,
	             initial_positions: np.ndarray,
	             model_based: bool,
	             movement_length: int,
	             resolution: int,
	             max_distance: float,
	             influence_radius: float,
	             forgetting_factor: float = 0.01,
				 reward_drone_type='weighted_idleness',
	             reward_type='weighted_importance',
	             reward_weights=(1, 1),
	             benchmark: str = 'algae_bloom',
	             model: str = 'miopic',
	             dynamic: bool = False,
	             seed: int = 0,
	             random_gt: bool = True,
	             int_observation: bool = False,
	             min_information_importance: float = 1.0,
	             previous_exploration=False,
	             pre_exploration_policy=None,
	             pre_exploration_steps=0,
				 camera_fov_angle: float = 160,						#
				 drone_height: float = 120,							#
				 n_drones: int = 1,									#
				 drone_direct_idleness_influece : bool = False,		#
				 blur_data: bool = False,							#
				 drone_noise: str = 'none',							#
				 fisheye_side: float = 1,							#
				 update_only_with_ASV: bool = False,				#		allows to set wheter the global idleness forgetting factor has to be added only with ASV read or with both the ASV and drone read
	             influence_drone_visited_map: bool = False,			#		allows to set wheter the visited map of the drones has to track only the effective position of the drone or to track the entire square of real influence 
				 influence_asv_visited_map: bool = False,			#		allows to set wheter the visited map of the asvs has to track only the effective position of the asv or to track the entire influence that the asv has
				 check_max_distances: bool = True					#		sets if the environment has to check and exclude the agents after they reach the max distance allowed
				 ):
		
		super().__init__(n_agents,
				 		navigation_map,
						initial_positions,
						model_based,
						movement_length,
						resolution,
						max_distance,
						influence_radius,
						forgetting_factor,
						reward_type,
						reward_weights,
						benchmark,
						model,
						dynamic,
						seed,
						random_gt,
						int_observation,
						min_information_importance,
						previous_exploration,
						pre_exploration_policy,
						pre_exploration_steps,
						check_max_distances)
		
		""" Copy attributes """
		self.n_drones = n_drones
		self.initial_air_positions = initial_air_positions
		self.max_air_distance = max_air_distance
		self.check_max_distances = check_max_distances
		#the influence_side has to be odd in order to have the drone always centered in a cell, the even values will be increased by 1
		influence_side = np.floor(influence_side)
		if influence_side % 2 == 0 :
			self.influence_side = influence_side + 1
		else :
			self.influence_side = influence_side
		
		self.forgetting_air_factor = forgetting_air_factor
		self.camera_fov_angle = camera_fov_angle
		self.drone_height = drone_height
		self.blur_data = blur_data
		self.drone_idleness_influence = drone_idleness_influence
		self.drone_direct_idleness_influence = drone_direct_idleness_influece
		self.reward_drone_type = reward_drone_type
		self.update_only_with_ASV = update_only_with_ASV
		self.influence_drone_visited_map = influence_drone_visited_map
		self.influence_asv_visited_map = influence_asv_visited_map

		""" Create the fleet """		
		self.fleet = CoordinatedHetFleet(n_vehicles = self.n_agents,							
										initial_surface_positions = self.initial_positions,		
										navigation_map = self.navigation_map,				
										total_max_surface_distance = self.max_distance,			
										influence_radius = self.influence_radius,					
										forgetting_factor = self.forgetting_factor,	
										air_forgetting_factor = self.forgetting_air_factor,	
										initial_air_positions = self.initial_air_positions,			
										total_max_air_distance = self.max_air_distance,				
										influence_side = self.influence_side,						
										camera_fov_angle = self.camera_fov_angle,					
										drone_height = self.drone_height,						
										blur_data = self.blur_data,	
										drone_idleness_influence = self.drone_idleness_influence,		
										drone_direct_idleness_influece = self.drone_direct_idleness_influence,					
										n_drones = self.n_drones,
										update_only_with_ASV=self.update_only_with_ASV,
										influence_drone_visited_map = influence_drone_visited_map,
										influence_asv_visited_map = influence_asv_visited_map,
										check_max_distances = self.check_max_distances)		

		#array of possible drone destinations
		self.possible_positions = np.argwhere(self.navigation_map == 1)

		""" Create the model """
		if model == 'miopic':
			self.model = HetMiopicModel(navigation_map=self.navigation_map,
			                      resolution=self.resolution,
			                      influence_radius=self.influence_radius,
			                      dt=0.01)
		elif model == 'none':
			self.model = HetMiopicModel(navigation_map=self.navigation_map,
			                         resolution=self.resolution,
			                         influence_radius=0,
			                         dt=0.7)
		elif model == 'vaeUnet':
			#the path of the model has to be written in the benchmark_2_vae_path dictionary in the UnetModel script
			self.model = VAEUnetDeepHetModel(navigation_map=self.navigation_map,
			                              model_path=benchmark_2_vae_path_het[benchmark],
			                              resolution=self.resolution,
			                              influence_radius=0.0, dt=0.01, N_imagined=10)
		else:
			raise ValueError('Unknown model')
		
		""" Create the Drone Reading Noise """
		if drone_noise == 'none':
			self.DroneNoiseModel = NoNoise()
		elif drone_noise == 'NoNoise':
			self.DroneNoiseModel = NoNoise()
		elif drone_noise == 'MeanNoise':
			self.DroneNoiseModel = MeanNoise()
		elif drone_noise == 'FishEyeNoise':
			self.DroneNoiseModel = FishEyeNoiseApproximator(self.influence_side, fisheye_side)

		#print(self.reward_type)

	def get_ASV_positions(self):
		return self.fleet.get_ASV_positions()
	
	def get_drone_positions(self):
		return self.fleet.get_drone_positions()
	
	def get_ASV_positions_dict(self):
		super().get_positions_dict()
	
	def get_drone_positions_dict(self):
		return {drone_id: position for drone_id, position in zip(self.fleet.drones_ids, self.fleet.get_drone_positions())}

	def reset(self):
		super().reset()

	def action_to_movement_ASV(self, action:int):
		super().action_to_movement(action)
	
	def random_ASV_action(self):
		super().random_action()

	"""
	private method to check if the position picked is already occupied by a drone
	"""
	def _check_drone_presence(self, position: np.array):
		for drone_id in self.fleet.drones_ids:
			if np.array_equal(self.fleet.drones[drone_id].position, position):
				return True
		return False 

	"""
	returns the size of the self.possible_positions - 1 which is also the maximum action number
	"""
	def max_action_id_drone(self):
		return len(self.possible_positions) - 1

	"""
	returns a random feasible position for the next destination of the drone
	"""
	def random_drone_action(self):

		random_i = np.random.randint(len(self.possible_positions))
		while self._check_drone_presence(self.possible_positions[random_i]):
			random_i = np.random.randint(len(self.possible_positions))
		
		return random_i

	"""
	returns the position which corresponds to the action number in input,
	we start to count action from the first cell to have a 1 value in top left corner and keep counting following the rows
	"""
	def action_to_position_drone(self, action: int):
		return (self.possible_positions[action])
	
	"""
	returns the number of the action knowing the destination position of the drone's next move
	"""
	def position_to_action_drone(self, position: np.array):
		for idx, element in enumerate(self.possible_positions):
			if(np.array_equal(position, element)):
				return idx
		return idx

	def get_feasible_positions(self):
		return self.possible_positions.copy()

	"""
	actions_ASV 	-> 	the old parameter it has the same meaning as the old one
	action_type_ASV -> 	the old parameter is action_type, it has the same meaning as the old one
	positions_Drone -> 	dictionary of id_drone:position where the new drones have to go
	ASV_moved 		-> 	flag that indicates if ASVs have acquired new data in the new step
	drone_moved 	-> 	flag that indicates if Drones have acquired new data in the new step
	"""
	def step(self, actions_ASV: dict, positions_Drone: dict, ASV_moved: bool, drone_moved: bool, action_type_ASV='discrete'):
		
		ASV_rewards = {}
		drone_rewards = {}
		
		#converts action integer number to movement (angle and length)
		if action_type_ASV == 'discrete':
			movements_orders = {key: self.action_to_movement(action) for key, action in actions_ASV.items()}
		else:
			movements_orders = actions
		

		if ASV_moved == True:
			# increase the ASV step counter by 1 unit
			self.steps += 1
			# Move the ASV fleet #
			self.fleet.move_ASVs(movements_orders, action_type = action_type_ASV)

		if drone_moved == True :
			# Move the Drone fleet #
			self.fleet.move_Drones(positions_Drone)

		
		self.fleet.end_step_movements()
		
		
		# Update the model #
		""" 
		the model used is shared between the two fleets, the air one and the water one
		even if the drone didn't make a move we still send all informations to the single update_model() method
		"""
		self.update_model(ASV_moved, drone_moved, positions_Drone)
		

		# Get the observations #
		""" the observations to be added into the new system are the ones created before:
		1) Navigation map with obstacles
		2) Positions of i agent
		3) Positions of the other agents
		4) Idleness
		5) Model

		6) Positions of i drone
		7) Positions of the other drones
		8) Air idleness

		even if the drone didn't make a move we still send all info, including the drone ones to the method caller
		"""
		observations = self.get_observations()
		

		# Get the rewards #
		"""
		two different reward functions:
		1) the ASVs reward function
		2) the Drones reward function 
		"""
		# Get ASV rewards #
		if ASV_moved == True :
			ASV_rewards = self.get_ASV_rewards()
		# Get drone rewards #
		if drone_moved == True :
			drone_rewards = self.get_drone_rewards()
		

		# Get the done flag #
		"""
		keeps the same method for everything since there is always data to lookup for both in the drone and the ASV
		"""
		done_ASV, done_Drone = self.get_done()
		
		# Get the information data #
		"""
		drone data has to be added into this method
		"""
		if self.eval:
			self.info = self.get_info()
		
		#if the dynamic flag is set calls for a step in the ground truth
		if self.dynamic:
			self.ground_truth.step()

		return observations, ASV_rewards, drone_rewards, done_ASV, done_Drone, self.info

	"""
	returns a dictionary with drone_id index, every drone_id has its own dictionary with all the square-cell-set having the center in every last waypoint position
	"""
	def _get_sample_drone_positions(self, positions: dict):
		offset = self.influence_side/2
		sample_drone_positions = {}
		for drone_id in positions.keys():

			position = positions[drone_id][0]
			
			column_start = int(np.ceil(position[1] - offset))
			row_start 	 = int(np.ceil(position[0] - offset))

			column_end 	 = int(np.floor(position[1] + offset))
			row_end 	 = int(np.floor(position[0] + offset))

			column_grid  = np.arange(column_start, column_end + 1)
			row_grid     = np.arange(row_start, row_end + 1)

			grid1, grid2 = np.meshgrid(row_grid, column_grid)

			sample_drone_positions[drone_id] = (np.column_stack((grid1.ravel(), grid2.ravel())))
		
		return sample_drone_positions

	# dafault model is Miopic #
	def update_model(self, ASV_moved: bool, drone_moved: bool, to_positions: dict):
		
		""" Save previous model """
		self.previous_model = self.model.predict().copy()

		sample_positions_ASV = []
		values_ASV = []
		""" if there is new data from the ASVs """
		if ASV_moved :
			sample_positions_ASV = self.fleet.get_last_ASV_waypoints()
			values_ASV = self.ground_truth.read(sample_positions_ASV)

		total_drone_positions_list = []
		total_values_drone = []
		""" if there is new data from the Drones """
		if drone_moved :
			Drone_positions_dict = self.fleet.get_last_drone_waypoints()
			Drone_squares_dict = self._get_sample_drone_positions(Drone_positions_dict)
			""" add the new drone data in the model for every drone """
			for drone_id in Drone_squares_dict.keys():
				if drone_id in to_positions:
					# Get the ground truth values #
					values_square = self.ground_truth.read(Drone_squares_dict[drone_id])
					# Apply the noise mask #
					drone_positions_list, values_Drone = self.DroneNoiseModel.mask(Drone_squares_dict[drone_id],values_square)
					total_drone_positions_list.append(drone_positions_list)
					total_values_drone.append(values_Drone)


		if self.model_str == 'deepUnet' or self.model_str == 'vaeUnet':
			self.model.update(from_ASV = ASV_moved, from_Drone = drone_moved, ASV_positions = sample_positions_ASV, ASV_values = values_ASV, Drone_positions = total_drone_positions_list, Drone_values = total_values_drone, ASV_visited_map = self.fleet.visited_map, Drone_visited_map = self.fleet.visited_air_map)
		else:
			self.model.update(from_ASV = ASV_moved, from_Drone = drone_moved, ASV_positions = sample_positions_ASV, ASV_values = values_ASV, Drone_positions = total_drone_positions_list, Drone_values = total_values_drone)
	
	def get_observations(self):
		""" Observation function. The observation is composed by:
		1) Navigation map with obstacles
		2) Positions of i agent
		3) Positions of the other agents
		4) Idleness
		5) Model
		"""
		self.ASV_observations = super().get_observations()
		self.Drone_observations = self._get_drone_observations()

	def _get_drone_observations(self):
		""" Observation function. The observation is composed by:
		6) Positions of i drone
		7) Positions of the other drones
		8) Air idleness
		"""
		# The observation is a dictionary of every active drone's observation #
		
		observations = {}
		
		#for every drone
		for drone_id in self.fleet.drones_ids:
			
			if self.int_observation:
				
				#saves the observation for that drone in the i-th position
				#convert the values to values in the [0,255] range
				observations[drone_id] = np.concatenate((
						# (255 * self.navigation_map[np.newaxis]).astype(np.uint8),
						(255 * self.fleet.get_drone_trajectory_map(observer=drone_id)[np.newaxis]).astype(
								np.uint8),
						(255 * self.fleet.get_drone_fleet_position_map(observers=drone_id)[np.newaxis]).astype(np.uint8),
						(255 * self.fleet.idleness_air_map[np.newaxis]).astype(np.uint8)
				), axis=0)
			
			else:
				
				observations[drone_id] = np.concatenate((
						# self.navigation_map[np.newaxis],
						self.fleet.get_drone_trajectory_map(observer=drone_id)[np.newaxis],
						self.fleet.get_drone_fleet_position_map(observers=drone_id)[np.newaxis],
						self.fleet.idleness_air_map[np.newaxis]
				), axis=0)
		
		return observations

	def get_ASV_rewards(self):
		return super().get_rewards()

	def get_drone_rewards(self):
		""" The reward is selected depending on the reward type """
		
		reward = {}
		if self.reward_drone_type == 'weighted_idleness':
			
			# Compute the reward as the local changes for the drone. #
			
			#it is a single value returned
			W_air = self.fleet.changes_air_idleness()  		# Compute the idleness air changes #
			
			for drone_id in self.fleet.drones_ids:
				# The reward is the sum of the idleness changes in the idleness_air which keeps track of both the asv and drone enviornment reads #
				
				#information gain calculated with idleness collected
				information_gain = np.sum(self.fleet.drones[drone_id].influence_mask * ( 	W_air[drone_id] / 
																							self.fleet.redundancy_drone_mask()	))
				#reward for every drone saved
				reward[drone_id] = information_gain * 100.0
		
		else:
			raise NotImplementedError('Unknown reward type')
		#print(reward)
		return reward

	def get_done(self):
		""" End the episode when the distance is greater than the max distance both for the drones and the ASVs """
			
		done_ASVs = super().get_done()

		done_Drones = { drone_id: False for drone_id in self.fleet.drones_ids }

		if self.check_max_distances:
			done_Drones = { drone_id: self.fleet.drones[drone_id].distance > self.max_air_distance for drone_id in self.fleet.drones_ids }

		return done_ASVs, done_Drones

	def get_info(self):
		""" This method returns the info of the step """
		
		y_real = self.ground_truth.read()[self.visitable_positions[:, 0], self.visitable_positions[:, 1]]
		y_pred = self.model.predict()[self.visitable_positions[:, 0], self.visitable_positions[:, 1]]
		
		#questo è l'errore medio tra una cella reale e la sua controparte rilevata dagli agenti
		mse_error = np.mean((y_real - y_pred) ** 2)
		mae_error = np.mean(np.abs(y_real - y_pred))
		normalization_value = np.sum(y_real)
		r2_score = 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2)

		total_average_ASV_distance = np.mean([veh.distance for veh in self.fleet.vehicles])
		mean_ASV_idleness = np.sum(self.fleet.idleness_map)/np.sum(self.navigation_map)
		mean_ASV_weighted_idleness = np.sum(self.fleet.idleness_map * self.ground_truth.read())/np.sum(self.navigation_map)
		coverage_ASV_percentage = np.sum(self.fleet.visited_map) / np.sum(self.navigation_map)
		if self.true_reward != {}:
			true_reward = np.sum([self.true_reward[agent_id] for agent_id in self.fleet.vehicles_ids])
		else:
			true_reward = 0

		total_average_drone_distance = np.mean([drn.distance for drn in self.fleet.drones])
		mean_air_idleness = np.sum(self.fleet.idleness_air_map)/np.sum(self.navigation_map)
		mean_air_weighted_idleness = np.sum(self.fleet.idleness_air_map * self.ground_truth.read())/np.sum(self.navigation_map)
		coverage_drone_percentage = np.sum(self.fleet.visited_air_map) / np.sum(self.navigation_map)
		
		return {'mse':                    			mse_error,
		        'mae':                    			mae_error,
		        'r2':                     			r2_score,
		        'total_average_ASV_distance': 		total_average_ASV_distance,
		        'mean_ASV_idleness':          		mean_ASV_idleness,
		        'mean_ASV_weighted_idleness': 		mean_ASV_weighted_idleness,
		        'coverage_ASV_percentage':    		coverage_ASV_percentage,
		        'normalization_value':    			normalization_value,
		        'true_reward':            			true_reward,
				'total_average_drone_distance':		total_average_drone_distance,
				'mean_air_idleness':				mean_air_idleness,
				'mean_air_weighted_idleness':		mean_air_weighted_idleness,
				'coverage_drone_percentage':		coverage_drone_percentage
		        }

	"""
	the discretemodelbasedpatrolling render includes:
	- navigation map
	- agent position (the method shows the first agent position)
	- fleet position (the other agents positions)
	- idleness map surface
	- the prediction of importance of the model
	- the ground truth

	the new method needs to add the new things introduced in the get_observations override:
	- Positions of i drone
	- Positions of the other drones
	- Air idleness
	"""
	def render(self):
		
		if self.fig is None:
			
			self.fig, self.axs = plt.subplots(3, 3)
			self.axs = self.axs.flatten()
			
			# ASV old class data #
			# Navigation map
			self.d0 = self.axs[0].imshow(self.navigation_map, cmap='gray', vmin=0, vmax=1)
			# self.d0 = self.axs[0].imshow(self.ground_truth.read(), cmap='gray', vmin=0, vmax=1)
			self.axs[0].set_title('Navigation map')
			# Agent position
			self.d1 = self.axs[1].imshow(self.ASV_observations[list(self.fleet.vehicles_ids)[0]][0], cmap='gray', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[1].set_title('Agent Position')
			# Fleet position
			self.d2 = self.axs[2].imshow(self.ASV_observations[list(self.fleet.vehicles_ids)[0]][1], cmap='gray', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[2].set_title('Fleet Position')
			# Idleness
			self.d3 = self.axs[3].imshow(self.ASV_observations[list(self.fleet.vehicles_ids)[0]][2], cmap='jet', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[3].set_title('Idleness')
			# Model
			self.d4 = self.axs[4].imshow(self.ASV_observations[list(self.fleet.vehicles_ids)[0]][3], cmap='jet', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[4].set_title('Model')
			# Ground truth
			self.d5 = self.axs[5].imshow(self.ground_truth.read(), cmap='jet', vmin=0, vmax=1)
			self.axs[5].set_title('Ground Truth')

			# New drone data #
			# Drone position
			self.d6 = self.axs[6].imshow(self.Drone_observations[list(self.fleet.drones_ids)[0]][0], cmap='gray', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[6].set_title('Drone position')
			# Air Idleness
			self.d7 = self.axs[7].imshow(self.Drone_observations[list(self.fleet.drones_ids)[0]][2], cmap='jet', vmin=0,
			                             vmax=1 if not self.int_observation else 255)
			self.axs[7].set_title('Global Idleness')

			# Drone fleet position, if there is more than one drone
			if self.n_drones > 1 :
				self.d8 = self.axs[8].imshow(self.Drone_observations[list(self.fleet.drones_ids)[0]][1], cmap='gray', vmin=0,
											vmax=1 if not self.int_observation else 255)
				self.axs[8].set_title('Fleet Position')
			
			plt.colorbar(self.d0, ax=self.axs[0])
		
		else:
			
			self.d0.set_data(self.navigation_map)
			# self.d0.set_data(self.ground_truth.read())
			self.d1.set_data(self.ASV_observations[list(self.fleet.vehicles_ids)[0]][0])
			self.d2.set_data(self.ASV_observations[list(self.fleet.vehicles_ids)[0]][1])
			self.d3.set_data(self.ASV_observations[list(self.fleet.vehicles_ids)[0]][2])
			self.d4.set_data(self.ASV_observations[list(self.fleet.vehicles_ids)[0]][3])
			self.d5.set_data(self.ground_truth.read())

			self.d6.set_data(self.Drone_observations[list(self.fleet.drones_ids)[0]][0])
			self.d7.set_data(self.Drone_observations[list(self.fleet.drones_ids)[0]][2])
			if self.n_drones > 1 :
				self.d8.set_data(self.Drone_observations[list(self.fleet.drones_ids)[0]][1])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		plt.pause(0.01)


if __name__ == "__main__":
	
	try:
		#swtich to a more interactive rendering backend for the plots
		plt.switch_backend('TkAgg')
		
		from PathPlanners.LawnMower import LawnMowerAgent
		from PathPlanners.NRRA import WanderingAgent
		import time
		
		scenario_map = np.genfromtxt('Maps/map.txt', delimiter=' ')
		
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
		                                   forgetting_factor=0.01,
		                                   max_distance=400,
		                                   benchmark='shekel',
		                                   dynamic=False,
		                                   reward_weights=[10, 10],
		                                   reward_type='weighted_idleness',
		                                   model='none',
		                                   seed=50000,
		                                   int_observation=True,
		                                   previous_exploration=True,
		                                   pre_exploration_steps=50,
		                                   pre_exploration_policy=preComputedExplorationPolicy("../PathPlanners/VRP/vrp_paths.pkl", n_agents=N),
		                                   )
		
		print(env.max_num_steps)
		env.eval = True
		for m in range(10):
			t0 = time.time()
			env.reset()
			#initializes the done array, with a flag for every agent
			done = {i: False for i in range(N)}
			
			mse = []
			rewards_list = []
			
			#generates the array of agents which follow the WanderingAgent model
			agent = {i: WanderingAgent(world=scenario_map, number_of_actions=8, movement_length=4, seed=0) for i in
			         range(N)}
			
			#while there is some agent which is not done yet
			while not all(done.values()):
				
				# actions = {i: np.random.randint(0,8) for i in done.keys() if not done[i]}
				#picks an action for every agent that is not yet done
				actions = {i: agent[i].move(env.fleet.vehicles[i].position.astype(int)) for i in done.keys() if
				           not done[i]}
				#executes the actions in the environment
				observations, rewards, done, info = env.step(actions)

				print("posizione agenti")
				print([env.fleet.vehicles[id].position for id in env.fleet.vehicles_ids])

				for id_pos,pos in {id: env.fleet.vehicles[id].position for id in env.fleet.vehicles_ids}.items():
					for id_pos2,pos2 in {id1: env.fleet.vehicles[id1].position for id1 in env.fleet.vehicles_ids if id1 != id_pos}.items():
						if np.array_equal(pos, pos2):
							print(pos, pos2)
							print("FOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\nFOUND\n")
				
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
			
			#prints the duration of the process
			print("Time: ", time.time() - t0)
			
			plt.close()
			plt.figure()
			print("Valore minimo mse -> " + str(min(mse)))
			plt.plot(mse)
			plt.show()
	
	
	
	
	except KeyboardInterrupt:
		print("Interrupted")
		plt.close()
