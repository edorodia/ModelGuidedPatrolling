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
from Models.GaussianProcessModel import GaussianProcessModel
from Models.PolinomialRegressor import PolinomialRegressor
from Models.SVRegressor import SVRegressor
from Models.UnetModel import UnetDeepModel, VAEUnetDeepModel, benchmark_2_vae_path
from sklearn.metrics import mean_squared_error, r2_score
from PathPlanners.dijkstra import Dijkstra

from Environment.exploration_policies import preComputedExplorationPolicy

from Environment.PatrollingEnvironment import *

if __name__ == "__main__":
		
	try:
		# OK #
		#swtich to a more interactive rendering backend for the plots
		plt.switch_backend('TkAgg')
		
		# ? #
		from PathPlanners.LawnMower import LawnMowerAgent
		from PathPlanners.NRRA import WanderingAgent
		import time
		
		# OK #
		scenario_map = np.genfromtxt('Maps/map.txt', delimiter=' ')
		
		# OK #
		N_ASV= 4
		N_drones = 1
		
		# OK #
		initial_ASV_positions = np.array([[42, 32],
									  [50, 40],
									  [43, 44],
									  [35, 45]])
		
		initial_drone_position = np.array([[16,24]])

		# OK #
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
					reward_type='weighted_importance',
					reward_weights=[10, 10],
					benchmark = 'shekel',
					model = 'none',
					dynamic = False,
					seed = 50000,
					int_observation = True,
					previous_exploration = False,
					pre_exploration_policy = None,
					pre_exploration_steps = 0, 
					camera_fov_angle = 160,						
					drone_height = 120,							
					n_drones = 1,									
					drone_direct_idleness_influece = False,		
					blur_data = False							
					)
		
		#print(env.max_num_steps)
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
			plt.plot(mse)
			plt.show()
	
	
	
	
	except KeyboardInterrupt:
		print("Interrupted")
		plt.close()