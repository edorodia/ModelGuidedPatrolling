import pickle

import matplotlib.pyplot as plt
import numpy as np
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import os


def positions_to_action(pos_1, pos_2):
	angle = np.arctan2(pos_2[1] - pos_1[1], pos_2[0] - pos_1[0])
	distance = np.linalg.norm(pos_2 - pos_1)
	
	return angle, distance


def tuple_to_dict(tup):
	dictionary = {'angle': tup[0], 'length': tup[1]}
	
	return dictionary


def main():
	RUNS = 100
	
	# Load the paths from pickle
	with open('PathPlanners/VRP/vrp_paths.pkl', 'rb') as handle:
		paths = pickle.load(handle)
	
	# Create the environment map
	scenario_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
	N = 4
	
	# Take the initial_positions from the paths
	initial_positions = np.asarray([paths[0][0], paths[1][0], paths[2][0], paths[3][0]])
	
	for i in range(N):
		x = np.arange(len(paths[i]))
		f = interp1d(x, paths[i], axis=0)
		xnew = np.arange(0, len(paths[i]) - 1, 0.5)
		paths[i] = f(xnew)
	
	
	# Remove the last element of the paths
	for i in range(N):
		paths[i] = paths[i][:-1]
	
	dataframe = []
	
	# Interpolate the paths to double the number of points

	
	for case in ['dynamic', 'static']:
		
		for benchmark in ['algae_bloom', 'shekel']:
			
			if benchmark == 'shekel' and case == 'dynamic':
				continue
		
			env = DiscreteModelBasedPatrolling(n_agents=N,
			                                   navigation_map=scenario_map,
			                                   initial_positions=initial_positions,
			                                   model_based=True,
			                                   movement_length=2,
			                                   resolution=1,
			                                   influence_radius=2,
			                                   forgetting_factor=0.5,
			                                   max_distance=600,
			                                   benchmark = benchmark,
			                                   dynamic = case == 'dynamic' and benchmark == 'algae_bloom',
			                                   reward_weights=[10, 10],
			                                   reward_type='weighted_idleness',
			                                   model='vaeUnet',
			                                   seed=50000,
			                                   int_observation=True,
			                                   )
			
			env.eval = True
			
			for run in tqdm(range(RUNS)):
				
				t = 0
				all_done = False
				env.reset()
				total_reward = 0
				
				while not all_done:
					# Take the first action
					actions = {i: paths[i][(t + 1) % len(paths[i])] for i in range(N)}
					
					# Take the step
					# Take the step
					obs, reward, done, info = env.step(actions, action_type='next_position')
					
					all_done = np.all(list(done.values()))
					
					total_reward += np.sum(list(reward.values()))
					
					# Render the environment
					# env.render()
					
					# plt.pause(0.5)
					
					t += 1
					
					dataframe.append([run, t, case, total_reward, info['mse'], info['mae'], info['r2'],
					                  info['total_average_distance'], info['mean_idleness'],
					                  info['mean_weighted_idleness'],
					                  info['coverage_percentage'], info['normalization_value'], 'VRP', benchmark])
	
	df = pd.DataFrame(dataframe,
	                  columns=['run', 'step', 'case', 'total_reward', 'mse', 'mae', 'r2', 'total_average_distance',
	                           'mean_idleness', 'mean_weighted_idleness', 'coverage_percentage',
	                           'normalization_value', 'Algorithm', 'Benchmark'])
	
	# Save the dataframe
	
	while True:
		
		res = input("do you want to append the results? (y/n) ")
		
		if res == 'y':
			df.to_csv('Evaluation/Patrolling/Results/vrp_results.csv', index=False, mode='a', header=False)
			break
		elif res == 'n':
			df.to_csv('Evaluation/Patrolling/Results/vrp_results.csv', index=False)
			break
		else:
			print('invalid input')


if __name__ == '__main__':
	
	try:
		main()
	except KeyboardInterrupt:
		print('Interrupted')
