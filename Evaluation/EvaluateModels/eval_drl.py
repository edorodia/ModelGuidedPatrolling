import pickle

import matplotlib.pyplot as plt
import numpy as np
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
from PathPlanners.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from tqdm import tqdm
import pandas as pd

plt.switch_backend('TkAgg')

render = False


def main():
	RUNS = 100
	
	# Create the environment map
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
	                                   benchmark='algae_bloom',
	                                   dynamic=False,
	                                   reward_weights=[10, 10],
	                                   reward_type='weighted_idleness',
	                                   model='vaeUnet',
	                                   seed=50000,
	                                   int_observation=True,
	                                   )
	
	multiagent = MultiAgentDuelingDQNAgent(env=env,
	                                       memory_size=int(1E3),
	                                       batch_size=64,
	                                       target_update=1000,
	                                       soft_update=True,
	                                       tau=0.001,
	                                       epsilon_values=[0.05, 0.05],
	                                       epsilon_interval=[0.0, 0.5],
	                                       learning_starts=100,
	                                       gamma=0.99,
	                                       lr=1e-4,
	                                       noisy=False,
	                                       train_every=50,
	                                       save_every=1000,
	                                       distributional=False,
	                                       masked_actions=True,
	                                       device='cuda:0',
	                                       logdir=None,
	                                       eval_episodes=10,
	                                       store_only_random_agent=True,
	                                       eval_every=1000)
	
	multiagent.dqn.eval()
	
	dataframe = []
	
	for reward_weight in ['i_1_w_0_1', 'i_1_w_1']:
		
		print('Reward weight: ', reward_weight)
		
		for case in ['dynamic', 'static']:
			print('Case: ', case)
			
			for benchmark in ['algae_bloom', 'shekel']:
				print('Benchmark: ', benchmark)
				
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
				                                   benchmark=benchmark,
				                                   dynamic=False,
				                                   reward_weights=[10, 10],
				                                   reward_type='weighted_idleness',
				                                   model='vaeUnet',
				                                   seed=50000,
				                                   int_observation=True,
				                                   )
				env.eval = True
				
				multiagent.env = env
				
				multiagent.load_model('runs/DRL/' + benchmark + '/' + reward_weight + '/' + 'FinalPolicy.pth')
				
				for run in tqdm(range(RUNS)):
					
					done = {agent_id: False for agent_id in range(multiagent.env.number_of_agents)}
					
					for module in multiagent.nogoback_masking_modules.values():
						module.reset()
					
					state = multiagent.env.reset()
					
					total_reward = 0
					t = 0
					
					while not all(done.values()):
						
						positions_dict = multiagent.env.get_positions_dict()
						actions = multiagent.select_masked_action(states=state, positions=positions_dict,
						                                          deterministic=True)
						
						actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}
						
						# Process the agent step #
						next_state, reward, done, info = multiagent.step(actions)
						
						if render:
							multiagent.env.render()
						
						# Update the state #
						state = next_state
						
						total_reward += np.sum(list(reward.values()))
						
						t += 1
						
						dataframe.append(
								[run, t, case, total_reward, info['true_reward'], info['mse'], info['mae'], info['r2'],
								 info['total_average_distance'], info['mean_idleness'],
								 info['mean_weighted_idleness'],
								 info['coverage_percentage'], info['normalization_value'], 'DRL_' + reward_weight, benchmark])
	
	df = pd.DataFrame(dataframe,
	                  columns=['run', 'step', 'case', 'total_reward', 'total_true_reward', 'mse', 'mae', 'r2',
	                           'total_average_distance',
	                           'mean_idleness', 'mean_weighted_idleness', 'coverage_percentage',
	                           'normalization_value', 'Algorithm', 'Benchmark'])
	
	# Save the dataframe
	
	while True:
		
		res = input("do you want to append the results? (y/n) ")
		
		if res == 'y':
			df.to_csv('Evaluation/Patrolling/Results/dlr_results.csv', index=False, mode='a', header=False)
			break
		elif res == 'n':
			df.to_csv('Evaluation/Patrolling/Results/dlr_results.csv', index=False)
			break
		else:
			print('invalid input')



if __name__ == '__main__':
	
	try:
		main()
	except KeyboardInterrupt:
		print('Interrupted')
