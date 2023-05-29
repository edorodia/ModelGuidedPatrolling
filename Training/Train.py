import sys
sys.path.append('.')

from PathPlanners.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
import numpy as np


map = np.genfromtxt('Environment\Maps\map.txt', delimiter=' ')

N = 4

initial_positions = np.array([[42,32],
                              [50,40],
                              [43,44],
                              [35,45]])

env = DiscreteModelBasedPatrolling(n_agents=N,
                                navigation_map=map,
                                initial_positions=initial_positions[:N],
                                model_based=True,
                                movement_length=3,
                                resolution=1,
                                influence_radius=3,
                                forgetting_factor=2,
                                max_distance=300,
                                benchmark='shekel',
                                dynamic=False,
                                reward_weights=[10.0, 100.0],
                                reward_type='local_changes',
                                model='miopic',
                                seed=5,)

multiagent = MultiAgentDuelingDQNAgent(env=env,
									memory_size=int(1E5),
									batch_size=128,
									target_update=1000,
									soft_update=True,
									tau=0.001,
									epsilon_values=[1.0, 0.05],
									epsilon_interval=[0.0, 0.33],
									learning_starts=100,
									gamma=0.99,
									lr=1e-4,
									noisy=False,
									train_every=15,
									save_every=5000,
									distributional=False,
									masked_actions=False,
									device='cuda:0',
				logdir=f'runs/Multiagent/Vehicles_4/',
				eval_episodes=10,
				eval_every=1000)

multiagent.train(episodes=10000)
