

import optuna

import pandas as pd

from optuna.importance import get_param_importances

import sys

sys.path.append('.')


import matplotlib.pyplot as plt

import pickle

import seaborn as sns


# Load study  using pickle

study_algae = joblib.load('runs/optuna/algae_bloom/20231022-215729/optuna_algae_bloom_20231023-041050')
study_shekel = joblib.load('runs/optuna/shekel/20231022-215721/optuna_shekel_20231023-050846')


# Print the best parameters for every study

print("algae bloom best parameters")
print(study_algae.best_params)


print("shekel best parameters")
print(study_shekel.best_params)

# Obtain the parameter importances

algae_importances = get_param_importances(study_algae)
shekel_importances = get_param_importances(study_shekel)

# From ordered dict to dataframe

algae_importances = algae_importances.items()
algae_importances = pd.DataFrame(algae_importances, columns = ['Parameter', 'Importance'])
# change name of parameters using the following dictionary
change_dict = {"L_perceptual": r"$\lambda_{percep}$", "L_KL_max": r"$\lambda_{KL}$", "L_reconstruction": r"$\lambda_{recons}$", "lr":"$lr$"}

algae_importances['Parameter'] = algae_importances['Parameter'].replace(change_dict)

shekel_importances = shekel_importances.items()
shekel_importances = pd.DataFrame(shekel_importances, columns = ['Parameter', 'Importance'])

shekel_importances['Parameter'] = shekel_importances['Parameter'].replace(change_dict)

# Plot the parameter importances in two 2x1 figure using seaborn and barplot

with sns.axes_style("darkgrid"):

    fig, axs = plt.subplots(2, 1, figsize = (10, 10), sharex=True)

    sns.barplot(x = 'Importance', y = 'Parameter', data = algae_importances, ax = axs[0], palette = 'Blues')
    sns.barplot(x = 'Importance', y = 'Parameter', data = shekel_importances, ax = axs[1], palette = 'Blues',)

    axs[0].set_title('Algae Bloom benchmark')
    axs[0].set_xlabel(None)
    axs[1].set_title('WQP benchmark')

    plt.show()

    # Plot the parameter importances

    print(algae_importances)
    print(shekel_importances)

# Print the selected parameters

print("algae bloom best parameters")
print(study_algae.best_params)

print("shekel best parameters")
print(study_shekel.best_params)