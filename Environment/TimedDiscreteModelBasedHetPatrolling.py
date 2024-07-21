import sys
sys.path.append("..")

from Environment.PatrollingEnvironment import DiscreteModelBasedHetPatrolling
import matplotlib.pyplot as plt
import numpy as np


"""
env:                        the DiscreteModelBasedHetPatrolling
speed_ratio:                tuple containing the speed ratio between drone and asv
							example: if the drone's speed is 100 km/h and the asv's speed is 25 km/h then the speed_ratio will be 4
							it has to be calculated with drone_speed/asv_speed
asv_path_planner:			object that generates the asv action to take
drone_path_planner:			object that generates the drone position where to move to
"""
class TimedDiscreteModelBasedHetPatrolling :
	def __init__(   self,
					env:          DiscreteModelBasedHetPatrolling,
					speed_ratio:  float,
					asv_path_planner,
					drone_path_planner,
					no_render: bool = False,
					no_print: bool = False):
		
		self.speed_ratio = speed_ratio
		self.env = env
		self.env.eval = True
		self.no_render = no_render
		self.no_print = no_print
		self.env.reset()
		
		self.n_agents = self.env.n_agents
		self.n_drones = self.env.n_drones

		#generate the path planners
		self.agent = {i: asv_path_planner(world=env.navigation_map, number_of_actions=8, movement_length=4, seed=0) for i in range(env.n_agents)}
		self.drone = {i: drone_path_planner(world=env.navigation_map) for i in range(env.n_drones)}

		self.current_time = 0.0
		self.movement_lenght = self.env.move_length

		#initialize the done arrays
		self.done_ASV = {i: False for i in range(self.n_agents)}
		self.done_Drone = {i: False for i in range(self.n_drones)}
		
		self.mse = []
		self.rewards_list = []

		#generate the first actions
		self.actions_ASV = {i: self.agent[i].move(self.env.fleet.vehicles[i].position.astype(int)) for i in self.done_ASV.keys() if
					not self.done_ASV[i]}
		self.positions_drone = {i: self.drone[i].move(self.env.fleet.drones[i].position.astype(int)) for i in self.done_Drone.keys() if
					not self.done_Drone[i]}
  
		distances = self._drone_distance(self.positions_drone)

		"""
			asv_actions 		: time
			drone1_position 	: time
			drone2_position 	: time
  		"""
		self.time_table = {tuple(["ASV" , frozenset(self.actions_ASV.items())]) : float(self.env.movement_length)}

  		#generate the time table
		actions_drone = {}
		for drone_id, distance in distances.items():
			self.time_table[tuple([drone_id, frozenset({drone_id : tuple(self.positions_drone[drone_id])}.items())])] = float(distance / self.speed_ratio) 
   
		plt.switch_backend('TkAgg')

		self.file = open("Timed_Simulator_Metrics.txt", "w")
		
	def check_ending(self):
		return (not all(self.done_ASV.values()) and not all(self.done_Drone.values()))

	#single environment step
	def step(self):

		self.file.write("Step :-\n")

		#find the minimum time value in the time table
		min_value = min(self.time_table.values())

		self.file.write(str(self.time_table) + "\n")

		list_actions = []

		#find all the other equal to the minimum values if there are some of them
		for action, time in self.time_table.items():
			if time == min_value:
				list_actions.append(action)
			self.time_table[action] = self.time_table[action] - min_value
			

		#for each element of the minimumns gather all the infos to make a global step at this time
		asv_moved = False
		drone_moved = False
		actions_ASV = {}
		positions_Drone = {}
		#select actions based of the time passed
		for action in list_actions:
			if action[0] == "ASV":
				self.file.write("ASV moved\n")
				#action to take is the move of the ASV fleet
				#gather the ASV action in the list of action to do in the step
				asv_moved = True
				actions_ASV = dict(action[1])

				#removes the selected ASV action from the time table
				del self.time_table[action]
				
			else:
				self.file.write("Drone moved -> " + str(dict(action[1])) + "\n" )
				#action to take is the move of a drone of the Drones fleet
				#gather the Drone action in the list of action to do in the step
				drone_moved = True
				positions_Drone.update(dict(action[1]))

				#removes the selected Drone action from the time table 
				del self.time_table[action]

				

		#executes the actions in the environment
		observations, ASV_rewards, drone_rewards, self.done_ASV, self.done_Drone, info = self.env.step( actions_ASV = actions_ASV, 
																								 		positions_Drone = positions_Drone,
																										ASV_moved = asv_moved, 
																										drone_moved = drone_moved)
		
		if asv_moved :
			#pick a new action and put it in the time table
			new_actions = {i: self.agent[i].move(self.env.fleet.vehicles[i].position.astype(int)) for i in self.done_ASV.keys() if 
				not self.done_ASV[i]}
			
			#puts the new action with the new time unit left in the time_table
			self.time_table[tuple(["ASV", frozenset(new_actions.items())])] = float(self.env.movement_length)

		if drone_moved :
			#pick a new position for the drone
			for key in positions_Drone.keys():
				i = key
				if self.done_Drone[i] == False :
					new_position = {i: self.drone[i].move(self.env.fleet.drones[i].position.astype(int))}

					#puts the new action with the new time unit left in the time table
					distance = np.linalg.norm(np.array(new_position[i]) - self.env.fleet.drones[i].position.astype(int))
					self.time_table[tuple([i, frozenset({i : tuple(new_position[i])}.items())])] = float(distance / self.speed_ratio)

		for i in range(self.n_agents):
				# If rewards dict does not contain the key, add it with 0 value #
				if i not in ASV_rewards.keys():
					ASV_rewards[i] = 0

		for i in range(self.n_drones):
			# If rewards dict does not contain the key, add it with 0 value #
			if i not in drone_rewards.keys():
				drone_rewards[i] = 0
		
		if self.no_render == False:
			self.env.render()
		
		if self.no_print == False:
			print("ASV_rewards: ", ASV_rewards)
			print("drone_rewards: ", drone_rewards)
			print("Done_ASV: ", self.done_ASV)
			print("Done_Drone: ", self.done_Drone)
			print("Info: ", info)
		
		#plt.pause(0.2)
		self.mse.append(info['mse'])

	"""
		runs an entire session of simulator till the drones and ASV reach the max distance
	"""
	def simulate_toEnd(self):
		counter = 0

		#keeps going till every drone and ASV is finished
		while not all(self.done_ASV.values()) and not all(self.done_Drone.values()):
			counter += 1
			self.step()
		
		self.file.write("Total steps in simulation :- " + str(counter))
		self.file.close()
		plt.close()
		plt.figure()
		print(str(self.done_ASV) + " " + str(self.done_Drone))
		print("Minimum mse -> " + str(min(self.mse)))
		plt.plot(self.mse)
		plt.show()

	"""
		calculates the euclidian distance for every new drones position generated
 	"""
	def _drone_distance(self, positions_drone: dict):
		return {i: np.linalg.norm(positions_drone[i] - self.env.fleet.drones[i].position) for i in positions_drone.keys()}
		

if __name__ == "__main__" : 
	try:
		from HetPathPlanners.RandomMover import RandomDroneMover, RandomVehicleMover
		import time
		
		scenario_map = np.genfromtxt('Maps/map.txt', delimiter=' ')
		
		N_ASV= 4
		N_drones = 1
		
		initial_ASV_positions = np.array([[42, 32],
									  [50, 40],
									  [43, 44],
									  [35, 45]])
		
		initial_drone_position = np.array([[16,24]])

		env = DiscreteModelBasedHetPatrolling( initial_air_positions = initial_drone_position,
					max_air_distance = 1000,
					influence_side = 9,
					forgetting_air_factor = 0.01,	
					drone_idleness_influence = 0.20,
					n_agents = N_ASV,
					navigation_map = scenario_map,
					initial_positions = initial_ASV_positions,
					model_based = True,
					movement_length = 2,
					resolution = 1,
					max_distance = 400,
					influence_radius = 2,
					forgetting_factor= 0.01,
					reward_drone_type='weighted_idleness',
					reward_type='weighted_idleness',
					reward_weights=[10, 10],
					benchmark = 'shekel',
					model = 'miopic',
					dynamic = False,
					seed = 50000,
					int_observation = True,
					previous_exploration = False,
					pre_exploration_policy = None,
					pre_exploration_steps = 0, 
					camera_fov_angle = 160,						
					drone_height = 120,							
					n_drones = N_drones,									
					drone_direct_idleness_influece = False,		
					blur_data = False,
					drone_noise = 'FishEyeNoise',
					fisheye_side=5,
					update_only_with_ASV=False
					)
		
		simulator = TimedDiscreteModelBasedHetPatrolling(	env = env,
															speed_ratio = 4.0,						#the drone is 4 times quicker than the ASV
															asv_path_planner = RandomVehicleMover,
															drone_path_planner = RandomDroneMover)

		simulator.simulate_toEnd()
	except KeyboardInterrupt:
		print("Interrupted")
		plt.close()


