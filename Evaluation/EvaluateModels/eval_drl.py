import pickle

import matplotlib.pyplot as plt
import numpy as np
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
from PathPlanners.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from tqdm import tqdm

plt.switch_backend('TkAgg')

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
                                   max_distance=600,
                                   benchmark='algae_bloom',
                                   dynamic=False,
                                   reward_weights=[10, 10],
                                   reward_type='weighted_idleness',
                                   model='vaeUnet',
                                   seed=500,
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
	
	multiagent.load_model('runs/DRL/algae_bloom/BestPolicy.pth')
	
	multiagent.dqn.eval()
			
	env.eval = True
	
	multiagent.env.reset()
	multiagent.env.reset()
	multiagent.env.reset()
	
	multiagent.evaluate_env(100, render=True, verbose=True)


if __name__ == '__main__':
	
	try:
		main()
	except KeyboardInterrupt:
		print('Interrupted')
