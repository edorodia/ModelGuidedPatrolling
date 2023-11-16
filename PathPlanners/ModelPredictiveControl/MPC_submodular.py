from abc import ABC
import sys

import matplotlib.pyplot as plt

sys.path.append('.')

import numpy as np
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
from tqdm import tqdm
import pandas as pd
from PathPlanners.ModelPredictiveControl.MCTS import MCTS, Node
import multiprocessing as mp
import os

MAX_TREE_LEVEL = 10
INFLUENCE_RADIUS = 2
POSSIBLE_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7]
PARALLEL = False
RENDER = True

if RENDER:
	plt.switch_backend('TkAgg')


class PatrollingNode(Node):
	
	def __init__(self, navigation_map, idleness_map, information_map, position, name, previous_action, reward, depth,
	             terminal=False):
		self.reward = reward
		self.position = position.copy()
		self.idleness_map = idleness_map.copy()
		self.information_map = information_map.copy()
		self.navigation_map = navigation_map.copy()
		self.terminal = terminal
		self.previous_action = previous_action
		self.name = name
		self.depth = depth
	
	def find_children(self):
		"""All possible successors of this state.
		Returns a set of PatrollingNode instances."""
		
		children = []
		
		for action in POSSIBLE_ACTIONS:
			# Iterate over all the possible actions and compute the new state #
			
			new_idleness_map, new_information_map, new_position, reward, is_valid = transition_function(
				self.idleness_map,
				self.information_map,
				self.navigation_map,
				self.position,
				action)
			
			if is_valid:
				children.append(PatrollingNode(navigation_map=self.navigation_map,
				                               idleness_map=new_idleness_map,
				                               information_map=new_information_map,
				                               position=new_position,
				                               reward=self.reward + reward,
				                               depth=self.depth + 1,
				                               previous_action=action,
				                               name=self.name + str(action)))
		
		return children
	
	def is_terminal(self):
		return self.terminal
	
	def get_reward(self):
		return self.reward
	
	def find_random_child(self):
		
		# Compute every single possible action #
		valid_mask = np.zeros(8)
		for action in POSSIBLE_ACTIONS:
			
			# Compute the new state #
			new_idleness_map, new_information_map, new_position, reward, is_valid = transition_function(
					self.idleness_map,
					self.information_map,
					self.navigation_map,
					self.position,
					action)
			
			if is_valid:
				valid_mask[action] = 1
		
		# Select a random action #
		action = np.random.choice(POSSIBLE_ACTIONS, p=valid_mask / np.sum(valid_mask))
		
		# Compute the new state #
		new_idleness_map, new_information_map, new_position, reward, is_valid = transition_function(self.idleness_map,
		                                                                                            self.information_map,
		                                                                                            self.navigation_map,
		                                                                                            self.position,
		                                                                                            action)
		
		return PatrollingNode(navigation_map=self.navigation_map,
		                      idleness_map=new_idleness_map,
		                      information_map=new_information_map,
		                      position=new_position,
		                      previous_action=action,
		                      reward=self.reward + reward,
		                      depth=self.depth + 1,
		                      name=self.name + str(action))
	
	def __hash__(self):
		return hash(self.name)
	
	def __eq__(self, other):
		return self.name == other.name
	
	def __repr__(self):
		return f"PatrollingNode({self.name})"
	
	def __str__(self):
		return f"PatrollingNode({self.name})"


def transition_function(idleness_map, information_map, navigation_map, position, action, forgetting_factor = 0.01):
	# Copy the objective map #
	new_idleness_map = idleness_map.copy()
	new_information_map = information_map.copy()
	
	# Compute the new position #
	movement = np.array([np.cos(2 * np.pi * action / 8).round().astype(int),
	                     np.sin(2 * np.pi * action / 8).round().astype(int)])
	
	next_attempt = np.clip(position + movement, 0, np.asarray(new_idleness_map.shape) - 1)
	
	if navigation_map[int(next_attempt[0]), int(next_attempt[1])] == 1:
		new_position = next_attempt.astype(int)
		is_valid_action = True
	else:
		new_position = position
		is_valid_action = False
	
	# Compute the reward #
	# 1) Compute the x and y coordinates of the circle
	x, y = np.meshgrid(np.arange(0, new_idleness_map.shape[1]), np.arange(0, new_idleness_map.shape[0]))
	x = x - position[1]
	y = y - position[0]
	
	distance = np.sqrt(x ** 2 + y ** 2)
	
	collected_information = np.sum(new_information_map[distance <= INFLUENCE_RADIUS])
	collected_idleness = np.sum(new_idleness_map[distance <= INFLUENCE_RADIUS])
	
	new_idleness_map += forgetting_factor
	new_idleness_map = np.clip(new_idleness_map, 0, 1)
	new_idleness_map[distance <= INFLUENCE_RADIUS] = 0.0
	
	reward = (1 + collected_information) * collected_idleness
	
	return new_idleness_map, new_information_map, new_position, reward, is_valid_action


def optimize_environment(environment):
	""" Optimize the environment using a Greedy approach """
	
	# Copy the navigation map #
	navigation_map = environment.navigation_map.copy()
	
	information_map = environment.model.predict().copy()
	idleness_map = environment.fleet.idleness_map.copy()
	
	# Copy the positions #
	positions = environment.get_positions_dict()
	
	# Action dictionary #
	actions = {agent_id: [] for agent_id in positions.keys()}
	
	for agent_id, agent_position in positions.items():
		
		tree = MCTS(exploration_weight=1.0, max_depth=MAX_TREE_LEVEL)
		
		# Create the root node #
		root = PatrollingNode(navigation_map=navigation_map,
		                      idleness_map=idleness_map,
		                      information_map=information_map,
		                      position=agent_position,
		                      name=f"agent_{agent_id}_position_{agent_position}_""",
		                      previous_action=None,
		                      depth=0,
		                      reward=0)
		
		for _ in range(50):
			tree.do_rollout(root)
		
		# Compute the best action #
		next_node = tree.choose(root)
		
		# print(f"Agent {agent_id} has chosen action {next_node.previous_action} with expected reward {
		# next_node.reward}")
		
		# Update the objective map for the next agent #
		idleness_map = next_node.idleness_map.copy()
		information_map = next_node.information_map.copy()
		
		# Select the action #
		actions[agent_id].append(next_node.previous_action)
	
	return actions


def experiment(arguments):
	RUNS = 100
	
	# Load the map
	nav_map = np.genfromtxt("Environment/Maps/map.txt", delimiter=' ')
	
	# Create the environment
	scenario_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
	
	N = 4
	control_horizon = 3
	
	# Take the initial_positions from the paths
	
	initial_positions = np.array([[42, 32],
	                              [50, 40],
	                              [43, 44],
	                              [35, 45]])
	
	dataframe = []
	
	benchmark = arguments['benchmark']
	case = arguments['case']
	
	print(f"Running benchmark {benchmark} with case {case}")
	
	env = DiscreteModelBasedPatrolling(n_agents=N,
	                                   navigation_map=scenario_map,
	                                   initial_positions=initial_positions,
	                                   model_based=True,
	                                   movement_length=2,
	                                   resolution=1,
	                                   influence_radius=2,
	                                   forgetting_factor=0.5,
	                                   max_distance=600,
	                                   benchmark=benchmark,
	                                   dynamic=case == 'dynamic',
	                                   reward_weights=[10, 10],
	                                   reward_type='weighted_idleness',
	                                   model='vaeUnet',
	                                   seed=50000,
	                                   int_observation=True,
	                                   min_information_importance=1.0,
	                                   )
	
	env.eval = True
	
	for run in tqdm(range(RUNS)):
		
		t = 0
		all_done = False
		env.reset()
		total_reward = 0
		
		while not all_done:
			# Optimize the environment
			
			best = optimize_environment(environment=env)
			
			next_action = {agent_id: best[agent_id][0] for agent_id in best.keys()}
			
			# Take the step
			obs, reward, done, info = env.step(next_action, action_type='discrete')
			
			total_reward += np.sum(list(reward.values()))
			
			all_done = np.all(list(done.values()))
			
			# Render the environment
			if RENDER:
				env.render()
			
			t += 1
			
			dataframe.append(
					[run, t, case, total_reward, info['true_reward'], info['mse'], info['mae'], info['r2'],
					 info['total_average_distance'], info['mean_idleness'],
					 info['mean_weighted_idleness'],
					 info['coverage_percentage'], info['normalization_value'], 'MCTS', benchmark])
	
	df = pd.DataFrame(dataframe,
	                  columns=['run', 'step', 'case', 'total_reward', 'total_true_reward', 'mse', 'mae', 'r2',
	                           'total_average_distance',
	                           'mean_idleness', 'mean_weighted_idleness', 'coverage_percentage',
	                           'normalization_value', 'Algorithm', 'Benchmark'])
	
	return df


if __name__ == "__main__":
	
	exp_args_0 = [{'benchmark': 'algae_bloom', 'case': 'static'},
	              {'benchmark': 'algae_bloom', 'case': 'dynamic'}]
	
	exp_args_1 = [{'benchmark': 'shekel', 'case': 'static'}]
	
	exp_args_all = exp_args_0 + exp_args_1
	
	try:
		
		if PARALLEL:
			pool = mp.Pool(processes=len(exp_args_all))
			results = pool.map(experiment, exp_args_all)
		else:
			results = []
			for exp_arg in exp_args_all:
				results.append(experiment(exp_arg))
		
		df = pd.concat(results, ignore_index=False)
		
		# Check if the folder exists
		if not os.path.exists('Evaluation/Patrolling/Results'):
			os.makedirs('Evaluation/Patrolling/Results')
		
		df.to_csv('Evaluation/Patrolling/Results/MCTS.csv', index=False)
		
		if PARALLEL:
			pool.join()
			pool.close()
	
	
	
	except KeyboardInterrupt:
		print("Exception occurred")
		print("Shutting down")
		pool.terminate()
