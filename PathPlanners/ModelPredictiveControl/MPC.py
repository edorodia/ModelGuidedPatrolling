import numpy as np
import time
from copy import copy
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
from multiprocessing import Pool
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from copy import copy, deepcopy


def compute_path_fitness(agent_actions_dict, positions, importance_map, influence_area_radius, length, navigation_map):
	""" Compute the sum of the importance map values for all agents given a set of points """
	
	fitness_value = 0
	
	for agent_id, agent_actions in agent_actions_dict.items():
		
		if agent_id not in positions.keys():
			continue
			
		# Initialize the agent positions
		agent_movements = length * np.asarray([[np.cos(2 * np.pi * action / 8).round().astype(int),
		                                        np.sin(2 * np.pi * action / 8).round().astype(int)] for action in
		                                       agent_actions])
		
		# Compute the fitness value for each agent
		
		position = positions[agent_id].astype(int)
		
		for movement in agent_movements:
			# Compute all the positions within the influence area circle
			if np.all(position + movement < importance_map.shape) and np.all(position + movement >= 0):
				if navigation_map[position[0] + movement[0], position[1] + movement[1]] == 1:
					position = position + movement
			
			# Compute the x and y coordinates of the circle
			x, y = np.meshgrid(np.arange(0, importance_map.shape[1]), np.arange(0, importance_map.shape[0]))
			x = x - position[1]
			y = y - position[0]
			
			# Compute the distance from the center of the circle
			distance = np.sqrt(x ** 2 + y ** 2)
			
			values = np.sum(importance_map[distance <= influence_area_radius])
			
			# Compute the fitness value
			fitness_value += values
			
			# Put to zero the importance map values within the circle
			importance_map[distance <= influence_area_radius] = 0.0
	
	return fitness_value


class Individual:
	
	def __init__(self, N_agents, max_length):
		self.gen = {indx: np.zeros((max_length,)) for indx in range(N_agents)}
		self.fitness = 0.0
		self.fitness_already_computed = False
	
	def __getitem__(self, item):
		return self.gen[item]
	
	def __setitem__(self, key, value):
		self.gen[key] = value
	
	def __str__(self):
		s = ""
		for indx, gen in self.gen.items():
			s += "Agent {} path: {}\n".format(indx, gen)
		
		return s
	
	def keys(self):
		return self.gen.keys()


def crossover_operation(parent1, parent2, crossover_probability):
	# Create two new children
	child1 = Individual(len(parent1.gen), len(parent1.gen[0]))
	child2 = Individual(len(parent1.gen), len(parent1.gen[0]))
	
	# Perform the crossover operation (2 points crossover) btween every agent inside the two parents
	for agent_id in parent1.keys():
		
		# Perform the crossover operation with a certain probability
		if np.random.rand() > crossover_probability:
			child1[agent_id] = parent1[agent_id].copy()
			child2[agent_id] = parent2[agent_id].copy()
			continue
		
		# Take the parents paths
		parent1_path = parent1[agent_id]
		parent2_path = parent2[agent_id]
		
		# Select the crossover points
		crossover_point1 = np.random.randint(0, len(parent1_path))
		crossover_point2 = np.random.randint(0, len(parent1_path))
		
		# Swap the paths
		if crossover_point1 > crossover_point2:
			crossover_point1, crossover_point2 = crossover_point2, crossover_point1
		
		# Perform the crossover operation
		child1[agent_id][:crossover_point1] = parent1_path[:crossover_point1]
		child1[agent_id][crossover_point1:crossover_point2] = parent2_path[crossover_point1:crossover_point2]
		child1[agent_id][crossover_point2:] = parent1_path[crossover_point2:]
		
		child2[agent_id][:crossover_point1] = parent2_path[:crossover_point1]
		child2[agent_id][crossover_point1:crossover_point2] = parent1_path[crossover_point1:crossover_point2]
		child2[agent_id][crossover_point2:] = parent2_path[crossover_point2:]
	
	child1.fitness_already_computed = False
	child2.fitness_already_computed = False
	
	return child1, child2


def mutation_operation(parent, mutation_probability):
	""" Change randomly the path of every agent with a certain probability """
	
	new_individual = Individual(len(parent.gen), len(parent.gen[0]))
	
	for agent_id in parent.keys():
		
		new_individual[agent_id] = parent[agent_id].copy()
		
		# Perform the mutation operation with a certain probability
		if np.random.rand() > mutation_probability:
			continue
		
		# Take the parent path
		new_individual_path = parent[agent_id].copy()
		
		# Select the mutation point
		mutation_point = np.random.randint(0, len(new_individual_path))
		
		# Change the path
		new_individual_path[mutation_point] = np.random.randint(0, 8)
		
		# Set the new path
		new_individual[agent_id] = new_individual_path
	
	return new_individual


def eval_population(population, environment, pool=None):
	""" Evaluate the fitness of the population """
	
	# Evaluate the fitness of each individual
	# Evaluate the fitness only if it is not already computed
	objective_map = environment.fleet.idleness_map * (0.1 + environment.model.predict())
	
	args = [(individual, environment, objective_map.copy()) for individual in population]
	new_population = []
	
	if pool is None:
		for individual in args:
			new_population.append(eval_individual(individual))
	else:
		for individual in pool.imap_unordered(eval_individual, args, chunksize=10):
			new_population.append(individual)
	
	return new_population


def eval_individual(args):
	# Evaluate the fitness of each individual
	# Evaluate the fitness only if it is not already computed
	
	individual = args[0]
	environment = args[1]
	objective_map = args[2]
	
	if not individual.fitness_already_computed:
		individual.fitness = compute_path_fitness(individual.gen, environment.get_positions_dict(),
		                                          objective_map.copy(),
		                                          environment.influence_radius, environment.movement_length,
		                                          environment.navigation_map)
		
		individual.fitness_already_computed = True
	
	return individual


def get_random_individuals(environment, horizon):
	# Create a new individual
	individual = Individual(environment.n_agents, horizon)
	
	# Create a random path for each agent
	for agent_id in range(environment.n_agents):
		individual[agent_id] = np.random.randint(0, 8, size=(horizon,))
	
	return individual


def mutate_and_cross(args):
	parent1, parent2, mutation_probability, crossover_probability = args
	
	# Perform the crossover operation
	child1, child2 = crossover_operation(parent1, parent2, crossover_probability)
	
	# Perform the mutation operation
	child1 = mutation_operation(child1, mutation_probability)
	child2 = mutation_operation(child2, mutation_probability)
	
	return child1, child2


def optimize_environment(environment, horizon, time_budget, verbose=0, pool=None):
	""" This function optimize the environment given the current agent positions and the importance map """
	
	# Genetic Algorithm parameters
	mu_population_size = 100
	lambda_population_size = 100
	mutation_probability = 0.2
	crossover_probability = 0.8
	
	# Create the initial population
	mu_population = [get_random_individuals(environment, horizon) for _ in range(mu_population_size)]
	
	# Start the optimization loop
	
	start_time = time.time()
	
	generation = 0
	
	best_fitness_hist = []
	
	best_individual = None
	
	# Mu + Lambda strategy
	while time.time() - start_time < time_budget:
		
		generation += 1
		
		# Evaluate the fitness of the population
		
		mu_population = eval_population(mu_population, environment, pool=pool)
		
		# Sort mu population by fitness
		mu_population.sort(key=lambda x: x.fitness, reverse=True)
		
		# Create the lambda population
		lambda_population = []
		
		# Create the new individuals
		# Use the pool if necessary
		
		# Select the parents
		args = [(mu_population[np.random.randint(0, mu_population_size)],
		         mu_population[np.random.randint(0, mu_population_size)],
		         mutation_probability, crossover_probability) for _ in range(lambda_population_size)]
		
		if pool is None:
			
			for parents in args:
				# Add the children to the lambda population
				children = mutate_and_cross(parents)
				lambda_population.append(children[0])
				lambda_population.append(children[1])
				
		else:
		
			for children in pool.imap_unordered(mutate_and_cross, args, chunksize=10):
				# Add the children to the lambda population
				lambda_population.append(children[0])
				lambda_population.append(children[1])
		
		# Evaluate the fitness of the population
		lambda_population = eval_population(lambda_population, environment, pool)
		
		# Sort the population by fitness
		lambda_population.sort(key=lambda x: x.fitness, reverse=True)
		
		# Select the best individuals
		mu_population = lambda_population[:mu_population_size]
		
		average_fitness = np.mean([individual.fitness for individual in mu_population])
		best_fitness = mu_population[0].fitness
		min_fitness = mu_population[-1].fitness
		
		if verbose > 0:
			print("GEN: %d -> Best fitness: %.3f | Average fitness: %.3f | Min fitness: %.3f" % (
					generation, best_fitness, average_fitness, min_fitness))
		
		best_fitness_hist.append(best_fitness)
		
		if best_fitness < np.median(best_fitness_hist[-10:]):
			break
			
		# Check if the best individual is better than the previous one
		if best_individual is None:
			best_individual = deepcopy(mu_population[0])
		elif mu_population[0].fitness > best_individual.fitness:
			best_individual = deepcopy(mu_population[0])
		
		

	# Return the best individual
	return best_individual


if __name__ == "__main__":
	
	pool = Pool(processes=4)
	
	print(f"Using {mp.cpu_count()} cpus")
	
	try:
		
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
		
		for case in ['dynamic', 'static']:
			
			env = DiscreteModelBasedPatrolling(n_agents=N,
			                                   navigation_map=scenario_map,
			                                   initial_positions=initial_positions,
			                                   model_based=True,
			                                   movement_length=2,
			                                   resolution=1,
			                                   influence_radius=2,
			                                   forgetting_factor=0.5,
			                                   max_distance=400,
			                                   benchmark='algae_bloom',
			                                   dynamic=case == 'dynamic',
			                                   reward_weights=[10, 10],
			                                   reward_type='local_changes', model='vaeUnet',
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
					# Optimize the environment
					if t % control_horizon == 0:
						best = optimize_environment(environment=env, horizon=20, time_budget=5, verbose=1, pool=pool)
						
					next_action = {agent_id: best[agent_id][0] for agent_id in range(N)}
					
					# Remove the first element of the paths
					for agent_id in range(N):
						best[agent_id] = best[agent_id][1:]
						
					# Take the step
					obs, reward, done, info = env.step(next_action, action_type='discrete')
					
					total_reward += np.sum(list(reward.values()))
					
					all_done = np.all(list(done.values()))
					
					# Render the environment
					# env.render()
					
					t += 1
					
					dataframe.append([run, t, case, total_reward, info['mse'], info['mae'], info['r2'],
					                  info['total_average_distance'], info['mean_idleness'],
					                  info['mean_weighted_idleness'],
					                  info['coverage_percentage'], info['normalization_value']])
		
		df = pd.DataFrame(dataframe,
		                  columns=['run', 'step', 'case', 'total_reward', 'mse', 'mae', 'r2', 'total_average_distance',
		                           'mean_idleness', 'mean_weighted_idleness', 'coverage_percentage',
		                           'normalization_value'])
		
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
		
		pool.terminate()
	
	
	except KeyboardInterrupt:
		print("Exception occurred")
		print("Shutting down")
		pool.terminate()
