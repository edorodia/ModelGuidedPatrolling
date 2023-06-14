import sys
sys.path.append('.')
from PathPlanners.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
import numpy as np
import time
import optuna
from optuna.trial import TrialState
import argparse
import joblib
import os

parser = argparse.ArgumentParser(description='Train a multiagent DQN agent to solve the patrolling problem.')
parser.add_argument('--benchmark', type=str, default='shekel', choices=['shekel', 'algae_bloom'], help='The benchmark to use.')
parser.add_argument('--seed', type=int, default=0, help='The seed to use.')
parser.add_argument('--n_agents', type=int, default=4, help='The number of agents to use.')
parser.add_argument('--movement_length', type=int, default=2, help='The movement length of the agents.')
parser.add_argument('--resolution', type=int, default=1, help='The resolution of the environment.')
parser.add_argument('--influence_radius', type=int, default=2, help='The influence radius of the agents.')
parser.add_argument('--forgetting_factor', type=int, default=2, help='The forgetting factor of the agents.')
parser.add_argument('--max_distance', type=int, default=300, help='The maximum distance of the agents.')
parser.add_argument('--w_reward_weight', type=float, default=1.0, help='The reward weights of the agents.')
parser.add_argument('--i_reward_weight', type=float, default=1.0, help='The reward weights of the agents.')
parser.add_argument('--model', type=str, default='miopic')
parser.add_argument('--device', type=str, default='cuda:0', help='The device to use.', choices=['cpu', 'cuda:0', 'cuda:1'])
parser.add_argument('--jobs', type=int, default=1, help='The device to use.')


# Compose a name for the experiment
args = parser.parse_args()

reward_weights = [args.w_reward_weight, args.i_reward_weight]

navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

N = 4

initial_positions = np.array([[42,32],
							  [50,40],
							  [43,44],
							  [35,45]])


def objective(trial: optuna.Trial):
    """ Optuna objective. """

    # Batch size
    batch_size = trial.suggest_int("batch_size", low = 4, high = 8, step=1)
    batch_size = 2 ** batch_size
    # Discount factor
    gamma = trial.suggest_float("gamma", low = 0.9, high = 0.999, step=0.001)
    # Tau 
    tau = trial.suggest_float("tau", low = 0.0001, high = 0.01, step=0.0001)
    # Epsilon values
    epsilon_values_final = trial.suggest_float("epsilon_final_value", low = 0.05, high = 0.2, step=0.05)
    # Epsilon interval
    epsilon_interval_final = trial.suggest_float("epsilon_final_interval", low = 0.25, high = 0.85, step=0.05)
    # Learning rate
    lr = trial.suggest_float("lr", low = 1e-5, high = 1e-3, step=1e-5)

    env = DiscreteModelBasedPatrolling(n_agents=N,
								navigation_map=navigation_map,
								initial_positions=initial_positions,
								model_based=True,
								movement_length=args.movement_length,
								resolution=1,
								influence_radius=args.movement_length,
								forgetting_factor=args.forgetting_factor,
								max_distance=args.max_distance,
								benchmark=args.benchmark,
								dynamic=False,
								reward_weights=reward_weights,
								reward_type='local_changes',
								model='miopic',
								seed=args.seed,)

    multiagent = MultiAgentDuelingDQNAgent(env=env,
                                        memory_size=int(1E6),
                                        batch_size=batch_size,
                                        target_update=1000,
                                        soft_update=True,
                                        tau=tau,
                                        epsilon_values=[1.0, epsilon_values_final],
                                        epsilon_interval=[0.0, epsilon_interval_final],
                                        learning_starts=100,
                                        gamma=gamma,
                                        lr=lr,
                                        noisy=False,
                                        train_every=15,
                                        save_every=None,
                                        distributional=False,
                                        masked_actions=True,
                                        device=args.device,
                                        logdir=None,
                                        eval_episodes=50,
                                        eval_every=None)

    for i in range(1, 10):
        
        # Train for 1000 episodes
        final_reward, _, _ = multiagent.train(episodes=1000, write_log=False, verbose=False)
        
        # Report the final reward as the objective value
        trial.report(final_reward, i)

        # Save the model with the study id
        multiagent.save_model(f'runs/DRL/Optuna/model_{study._study_id}.pth')

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return final_reward	


if __name__ == "__main__":

    # Create a directory for the study
    if not os.path.exists('runs/DRL/Optuna'):
        os.mkdir(f'runs/DRL/Optuna')

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(), study_name="DQN_hyperparametrization")

    study.optimize(objective, n_trials=40, n_jobs=args.jobs)

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
        print("    {}: {}".format(key, value))

    joblib.dump(study, "study.pkl")