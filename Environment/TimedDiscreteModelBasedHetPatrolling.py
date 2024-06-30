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
					env:                       DiscreteModelBasedHetPatrolling,
					speed_ratio:  tuple[int, int],
					asv_path_planner,
					drone_path_planner):
		
		self.speed_ratio = speed_ratio
		self.env = env
		
		self.n_agents = self.env.n_agents
		self.n_drones = self.env.n_drones

		#generates the agents acton path planner
		self.agent = {i: asv_path_planner(world=env.navigation_map, number_of_actions=8, movement_length=4, seed=0) for i in
					 range(env.n_agents)}
		
		#generates the drones action path planner
		self.drone = {i: drone_path_planner(world=env.navigation_map) for i in range(env.n_drones)}

		self.current_time = 0.0

		#initializes the done array, with a flag for every agent
		done_ASV = {i: False for i in range(self.n_agents)}
		#initializes the done array, with a flag for every drone
		done_Drone = {i: False for i in range(self.n_drones)}
		
		self.mse = []
		self.rewards_list = []

		plt.switch_backend('TkAgg')

	def step(self):
		#if the time is the beginning generate a new action for every agent and drone in the environment
		#picks an action for every vehicle that is not yet done
		actions_ASV = {i: self.agent[i].move(self.env.fleet.vehicles[i].position.astype(int)) for i in self.done_ASV.keys() if
					not self.done_ASV[i]}

		#picks an action for every drone that is not yet done
		positions_drone = {i: self.drone[i].move(self.env.fleet.drones[i].position.astype(int)) for i in self.done_Drone.keys() if
					not self.done_Drone[i]}
		
		#executes the actions in the environment, it has to execute them in the right moment depending on the quickness of the drone
		observations, ASV_rewards, drone_rewards, done_ASV, done_Drone, info = self.env.step(actions_ASV, positions_drone, True, True)

		




