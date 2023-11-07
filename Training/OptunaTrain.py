import optuna
from optuna.trial import TrialState
import os
import joblib
from PathPlanners.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='Train a multiagent DQN agent to solve the patrolling problem.')
parser.add_argument('--benchmark', type=str, default='shekel', choices=['shekel', 'algae_bloom'], help='The benchmark to use.')
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

navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

N = 4

initial_positions = np.array([[42, 32],
							  [50, 40],
							  [43, 44],
							  [35, 45]])

device = 'cpu' if args.device == -1 else f'cuda:{args.device}'


def objective(trial: optuna.Trial):
	""" Optuna objective. """

	# Batch size
	batch_size = trial.suggest_int("batch_size", low = 4, high = 8, step=1)
	batch_size = 2 ** batch_size
	# Discount factor
	gamma = trial.suggest_float("gamma", low = 0.9, high = 0.999, step=0.001)

	# Epsilon values
	epsilon_values_final = trial.suggest_float("epsilon_final_value", low = 0.05, high = 0.2, step=0.05)
	# Epsilon interval
	epsilon_interval_final = trial.suggest_float("epsilon_final_interval", low = 0.25, high = 0.85, step=0.05)
	# Learning rate
	lr = trial.suggest_float("lr", low = 1e-5, high = 1e-3, step=1e-5)

	experiment_name = "DRL_OPTUNA_TRIAL_" + str(trial.value)

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
										   batch_size=batch_size,
										   target_update=1000,
										   soft_update=True,
										   tau=0.001,
										   epsilon_values=[1.0, epsilon_values_final],
										   epsilon_interval=[0.0, epsilon_interval_final],
										   learning_starts=50,
										   gamma=gamma,
										   lr=lr,
										   noisy=False,
										   train_every=50,
										   save_every=5000,
										   distributional=False,
										   masked_actions=True,
										   device=device,
										   logdir=f'runs/DRL/Optuna/{experiment_name}',
										   eval_episodes=50,
										   store_only_random_agent=True,
										   eval_every=10000)

	total_reward = 0.0

	for train_step in range(10):
		
		multiagent.train(episodes=100)
	
		total_reward, _, _ = multiagent.evaluate_env(20)
	
		# Report the trial
		study.report(total_reward)
	
		# Handle pruning based on the intermediate value.
		if study.should_prune():
			raise optuna.TrialPruned()

	return total_reward


if __name__ == "__main__":

	# Create a directory for the study
	if not os.path.exists('runs/DRL/Optuna'):
		os.mkdir(f'runs/DRL/Optuna')

	study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(), study_name="DQN_hyperparametrization")

	study.optimize(objective, n_trials=40)

	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("	{}: {}".format(key, value))

	joblib.dump(study, "runs/DRL/Optuna/DRL_hyperparam_study.pkl")