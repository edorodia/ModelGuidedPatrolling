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
from sklearn.metrics import mean_squared_error, r2_score
from PathPlanners.dijkstra import Dijkstra

from Environment.exploration_policies import preComputedExplorationPolicy

from Environment.PatrollingEnvironment import *

if __name__ == "__main__":
		
	try:
		# OK #
		#swtich to a more interactive rendering backend for the plots
		plt.switch_backend('TkAgg')

		secondfile = open("Metrics_NotTimedSimulation.txt", "w")

		counter = 0
		
		# OK #
		from HetPathPlanners.RandomMover import RandomDroneMover, RandomVehicleMover
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
		
		#print(env.max_num_steps)
		env.eval = True
		for m in range(10):
			t0 = time.time()
			env.reset()
			#initializes the done array, with a flag for every agent
			done_ASV = {i: False for i in range(N_ASV)}
			#initializes the done array, with a flag for every drone
			done_Drone = {i: False for i in range(N_drones)}
			
			mse = []
			rewards_list = []
			
			#generates the array of agents which follow the RandomVehicleMover model
			agent = {i: RandomVehicleMover(world=scenario_map, number_of_actions=8, movement_length=4, seed=0) for i in
					 range(N_ASV)}
					 
			#generates the array of agents which follow the RandomDroneMover model
			drone = {i: RandomDroneMover(world=scenario_map) for i in range(N_drones)}
			print(drone)
			#while there is some agent which is not done yet
			while not all(done_ASV.values()) and not all(done_Drone.values()):

				counter += 1

				#picks an action for every vehicle that is not yet done
				actions_ASV = {i: agent[i].move(env.fleet.vehicles[i].position.astype(int)) for i in done_ASV.keys() if
						   not done_ASV[i]}

				secondfile.write("ASV moved -> " + str(actions_ASV) + "\n")

				#picks an action for every drone that is not yet done
				positions_drone = {i: drone[i].move(env.fleet.drones[i].position.astype(int)) for i in done_Drone.keys() if
						   not done_Drone[i]}
				
				secondfile.write("Drone moved -> " + str(positions_drone) + "\n")
				
				#executes the actions in the environment, it has to execute them in the right moment depending on the quickness of the drone
				observations, ASV_rewards, drone_rewards, done_ASV, done_Drone, info = env.step(actions_ASV, positions_drone, True, True)
				
				for i in range(N_ASV):
					# If rewards dict does not contain the key, add it with 0 value #
					if i not in ASV_rewards.keys():
						ASV_rewards[i] = 0
				
				rewards_list.append([ASV_rewards[i] for i in range(N_ASV)])

				for i in range(N_drones):
					# If rewards dict does not contain the key, add it with 0 value #
					if i not in drone_rewards.keys():
						drone_rewards[i] = 0
				
				rewards_list.append([drone_rewards[i] for i in range(N_drones)])
				
				env.render()
				
				print("ASV_rewards: ", ASV_rewards)
				print("drone_rewards: ", drone_rewards)
				print("Done_ASV: ", done_ASV)
				print("Done_Drone: ", done_Drone)
				print("Info: ", info)
				
				plt.pause(0.2)
				mse.append(info['mse'])
			
			#prints the duration of the process
			print("Time: ", time.time() - t0)

			secondfile.write("Total steps in the simulation :- " + str(counter))
			secondfile.close()

			plt.close()
			plt.figure()
			print("Valore minimo mse -> " + str(min(mse)))
			plt.plot(mse)
			plt.show()
	
	
	
	
	except KeyboardInterrupt:
		print("Interrupted")
		plt.close()