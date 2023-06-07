import sys
sys.path.append('.')

from PathPlanners.NRRA import WanderingAgent
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing as mp
import deap
import signal

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
from deap.algorithms import eaMuPlusLambda
import seaborn as sns
import pandas as pd
import pickle

def init_pool():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

""" Create an environment """

nav_map = np.genfromtxt('Environment\Maps\map.txt', delimiter=' ')
N = 4

initial_positions = np.array([[42,32],
                            [50,40],
                            [43,44],
                            [35,45]])

env = DiscreteModelBasedPatrolling(n_agents=N,
								navigation_map=nav_map,
								initial_positions=initial_positions,
								model_based=True,
								movement_length=2,
								resolution=1,
								influence_radius=2,
								forgetting_factor=2,
								max_distance=200,
								benchmark='shekel',
								dynamic=False,
								reward_weights=[10.0, 100.0],
								reward_type='local_changes',
								model='miopic',
								seed=5000,
                                random_gt=False)

""" Create a genetic algorithm """

iteration = 0
seed = 0

plt.imshow(env.ground_truth.read())
plt.show()

# Create a fitness function
def fitness_function(individual, render=False):

    # Transform the individual numpy array into a TxN array of actions
    individual = np.array(individual).reshape(-1,N)

    # Reset the environment

    W_mean = []
    I_mean = []

    path = []

    for _ in range(1):

        env.reset()

        path.append(env.fleet.get_positions().copy())

        # Run the environment
        W = 0.0
        I = 0.0

        for t in range(individual.shape[0]):
            # Get the actions of the agents at time t
            actions = {i:individual[t,i] for i in range(N)}

            # Step the environment
            _, _, _, info = env.step(actions)
            path.append(env.fleet.get_positions().copy())
            if render:
                env.render()    

            W += info['W']
            I += info['I']

        W_mean.append(W.copy())
        I_mean.append(I.copy())

    # Set the seed to an arbitrary value

    return np.mean(W_mean), np.mean(I_mean), np.asarray(path)


# Create a toolbox
toolbox = base.Toolbox()

# Create a creator for the individuals and register 
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
toolbox.register("attr_bool", np.random.randint, 0, 8)
toolbox.register("individual", np.random.randint, 0, 8, N*100)


# Register the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


if __name__ == '__main__':

    # Read the deap checkpointer

    cp = pickle.load(open('SomeOthersScripts/optimization.pkl', 'rb'))

    # Select one individual from the pareto front

    hof = cp['halloffame']

    # Get the fitness values of the individuals in the pareto front 
    W_pareto = np.asarray([ind.fitness.values[0] for ind in hof])
    I_pareto = np.asarray([ind.fitness.values[1] for ind in hof])

    # Create a dataframe
    df_pareto = pd.DataFrame({'W objective':W_pareto, 'I objective':I_pareto})
    sns.set_theme(style="whitegrid")

    # Plot the pareto front
    sns.scatterplot(data=df_pareto, x='W objective', y='I objective')
    #Rename the axes
    plt.xlabel(r'$\mathcal{W}$ objective')
    plt.ylabel(r'$\mathcal{I}$ objective')

    _,_, path_0 = fitness_function(hof[0])
    _,_, path_1 = fitness_function(hof[-1])

    # Plot the two extremes of the pareto front
    plt.scatter(W_pareto[0], I_pareto[0], marker='x', color='red', s=100, label='W-Greedy')
    plt.scatter(W_pareto[-1], I_pareto[-1], marker='x', color='green', s=100, label='I-Greedy')

    plt.legend()

    plt.tight_layout()
    plt.show()

    nan_mask = nav_map.copy()
    nan_mask[nan_mask == 0] = np.nan

    # Plot the two paths over the ground_Truth
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    # Erase the grids
    for idx in range(2):
        ax[idx].grid(False)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        # Erase the box
        for spine in ax[idx].spines.values():
            spine.set_visible(False)

    cmap = matplotlib.cm.viridis
    cmap.set_bad('white', 1.)
    colors = ['red', 'orange', 'purple', 'green']
    ax[0].imshow(env.ground_truth.read()*nan_mask, cmap=cmap)
    for idx in range(N):
        ax[0].plot(path_0[:,idx, 1], path_0[:,idx, 0], color=colors[idx], linewidth=2, marker='o', markersize=5)

    ax[1].imshow(env.ground_truth.read()*nan_mask, cmap=cmap)
    for idx in range(N):
        ax[1].plot(path_1[:,idx, 1], path_1[:,idx, 0], color=colors[idx], linewidth=2, marker='o', markersize=5)

    plt.tight_layout()
    plt.show()











