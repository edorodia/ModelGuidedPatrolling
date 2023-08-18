import sys
sys.path.append('.')

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Set the style
sns.set_style('whitegrid')
sns.set_palette('deep')

sns.set_theme()


# Load the data
data = pd.read_csv('Evaluation/EvaluateModels/Results/results_all.csv')

# Accumulate time for each run along the steps
data['acc_time'] = data.groupby(['benchmark', 'model', 'run'])['time'].cumsum()


# Plot the average RUN results with X = step and Y = mse


for benchmark in data['benchmark'].unique():

    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    benchmark_str = 'Algae Bloom benchmark (dynamic case)' if benchmark == 'algae_bloom' else 'Shekel benchmark'

    sns.lineplot(data=data[data['benchmark'] == benchmark], x='step', y='rmse', hue='model', style='model', lw=2, ax=axs[0])
    axs[0].set_title(benchmark_str)

    sns.lineplot(data=data[data['benchmark'] == benchmark], x='step', y='weighted_rmse', hue='model', style='model', lw=2, ax=axs[1])

    axs[0].set_ylabel('RMSE')
    axs[1].set_ylabel('Weighted RMSE')
    axs[1].set_xlabel('Step')

    axs[0].legend(loc = 'best', ncols = 2)
    axs[1].get_legend().remove()

    axs[0].set_xlim([0, 100])

    for model in data['model'].unique():
        subdata = data[(data['benchmark'] == benchmark) & (data['model'] == model)]
        print(f"****** Model: {model} | Benchmark: {benchmark} ******")
        print(f"RMSE (33%): {subdata[subdata['step'] == 32]['rmse'].mean()} Weighted RMSE (33%): {subdata[subdata['step'] == 32]['weighted_rmse'].mean()}")
        print(f"RMSE (66%): {subdata[subdata['step'] == 62]['rmse'].mean()} Weighted RMSE (66%): {subdata[subdata['step'] == 62]['weighted_rmse'].mean()}")
        print(f"RMSE (100%): {subdata[subdata['step'] == 99]['rmse'].mean()} Weighted RMSE (100%): {subdata[subdata['step'] == 99]['weighted_rmse'].mean()}")

    plt.tight_layout()
    plt.show()


sns.set_palette('deep')
# Box plot the average time for each model and benchmark 
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True)

sns.boxplot(data=data, x='model', y='time', ax=axs, showfliers=False, order = ['gp', 'vaeUnet', 'deepUnet', 'knn', 'miopic'])
plt.yscale('log')
#sns.stripplot(x="model", y="time", data=data,size=4, color=".3", linewidth=0, ax=axs, order = ['gp', 'vaeUnet', 'deepUnet', 'knn', 'miopic'])

axs.set_ylabel('Time [s]')
axs.set_xlabel(None)

plt.tight_layout()
plt.show()





for benchmark in data['benchmark'].unique():
    # Plot, for every model, the model_map and the gt_map #
    RUN = 0
    fig, axes = plt.subplots(1, 1 + len(data['model'].unique()), figsize=(8, 4))
    # Get the maps
    model_maps = {model: np.load(f'Evaluation/EvaluateModels/Results/estimated_model_{benchmark}_{model}.npy') for model in data['model'].unique()}
    gt_maps = {model: np.load(f'Evaluation/EvaluateModels/Results/gts_evaluation_{benchmark}_{model}.npy') for model in data['model'].unique()}
    colors = ['red', 'orange', 'green', 'yellow', 'yellow', 'black', 'yellow', 'pink', 'brown', 'gray']
    positions = {model: np.load(f'Evaluation/EvaluateModels/Results/positions_{benchmark}_{model}.npy') for model in data['model'].unique()}
    mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
    mask[mask == 0] = np.nan


    for ix, model in enumerate(data['model'].unique()):

        axes[ix].imshow(model_maps[model][RUN][-1] * mask, cmap = 'coolwarm', vmin = 0, vmax = 1, zorder=10)
        axes[ix].set_title(model)
        axes[ix].set_xticks([])
        axes[ix].set_yticks([])

        model_positions = positions[model][RUN]

        # Plot the positions
        for agent_id in range(model_positions.shape[1]):
            axes[ix].plot(model_positions[:, agent_id, 1], model_positions[ :, agent_id, 0], linestyle = '-', alpha = 0.3, marker='o', markersize=2, color=colors[agent_id], label=f'Agent {agent_id}' if ix == 0 else None, zorder=60)

        if ix == 0:
            axes[ix].legend().set_zorder(100)

    im = axes[-1].imshow(gt_maps[model][RUN]*mask, cmap = 'coolwarm', vmin = 0, vmax = 1, zorder=10)
    axes[-1].set_title('Ground Truth')
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    plt.show()