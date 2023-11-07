import sys

sys.path.append('.')
from PathPlanners.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
import numpy as np
import time

import argparse

parser = argparse.ArgumentParser(description='Train a multiagent DQN agent to solve the patrolling problem.')

parser.add_argument('--benchmark', type=str, default='shekel', choices=['shekel', 'algae_bloom'],
                    help='The benchmark to use.')

parser.add_argument('--seed', type=int, default=0, help='The seed to use.')
parser.add_argument('--n_agents', type=int, default=4, help='The number of agents to use.')
parser.add_argument('--movement_length', type=int, default=2, help='The movement length of the agents.')
parser.add_argument('--resolution', type=int, default=1, help='The resolution of the environment.')
parser.add_argument('--influence_radius', type=int, default=2, help='The influence radius of the agents.')
parser.add_argument('--forgetting_factor', type=int, default=0.5, help='The forgetting factor of the agents.')
parser.add_argument('--max_distance', type=int, default=400, help='The maximum distance of the agents.')
parser.add_argument('--model', type=str, default='vaeUnet', choices=['miopic', 'vaeUnet'], help='The model to use.')
parser.add_argument('--device', type=int, default=0, help='The device to use.', choices=[-1, 0, 1])
parser.add_argument('--dynamic', type=bool, default=False, help='Simulate dynamic')

# Compose a name for the experiment
args = parser.parse_args()

if not args.dynamic:
	experiment_name = f'Experiment_benchmark_{args.benchmark}_' + 'dynamic' if args.dynamic else 'static' + f'_model_{args.model}_{time.strftime("%Y%m%d-%H%M%S")}'

navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

N = 4

initial_positions = np.array([[42, 32],
                              [50, 40],
                              [43, 44],
                              [35, 45]])

device = 'cpu' if args.device == -1 else f'cuda:{args.device}'


env = DiscreteModelBasedPatrolling(n_agents=N,
                                   navigation_map=navigation_map,
                                   initial_positions=initial_positions,
                                   model_based=True,
                                   movement_length=args.movement_length,
                                   resolution=args.resolution,
                                   influence_radius=args.influence_radius,
                                   forgetting_factor=args.forgetting_factor,
                                   max_distance=args.max_distance,
                                   benchmark='algae_bloom',
                                   dynamic=args.dynamic,
                                   reward_type='weighted_idleness',
                                   model='vaeUnet',
                                   seed=50000,
                                   int_observation=True,
                                   )

multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E6),
                                       batch_size=64,
                                       target_update=1000,
                                       soft_update=True,
                                       tau=0.001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.5],
                                       learning_starts=100,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=False,
                                       train_every=50,
                                       save_every=5000,
                                       distributional=False,
                                       masked_actions=True,
                                       device=device,
                                       logdir=f'runs/DRL/{experiment_name}',
                                       eval_episodes=10,
                                       store_only_random_agent = True,
                                       eval_every=1000)

multiagent.train(episodes=10000)
