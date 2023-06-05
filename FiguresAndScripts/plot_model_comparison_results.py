import sys
sys.path.append('.')

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set the style
sns.set_style('whitegrid')


# Load the data
data = pd.read_csv('Evaluation/EvaluateModels/Results/results_all.csv')

# Accumulate time for each run along the steps
data['time'] = data.groupby(['benchmark', 'model', 'run'])['time'].cumsum()


# Plot the average RUN results with X = step and Y = mse

sns.lineplot(data=data[data['benchmark'] == 'shekel'], x='step', y='rmse', hue='model')
plt.show()

sns.lineplot(data=data[data['benchmark'] == 'algae_bloom'], x='step', y='rmse', hue='model')
plt.show()



mask = np.genfromtxt('Environment/Maps/map_large.txt', delimiter=' ')

for benchmark in data['benchmark'].unique():
    # Plot, for every model, the model_map and the gt_map #
    RUN = 3
    fig, axes = plt.subplots(1, 1 + len(data['model'].unique()), figsize=(8, 4))
    # Get the maps
    model_maps = {model: np.load(f'Evaluation/EvaluateModels/Results/estimated_model_{benchmark}_{model}.npy') for model in data['model'].unique()}
    gt_maps = {model: np.load(f'Evaluation/EvaluateModels/Results/gts_evaluation_{benchmark}_{model}.npy') for model in data['model'].unique()}
    colors = ['red', 'orange', 'green', 'yellow', 'yellow', 'black', 'yellow', 'pink', 'brown', 'gray']
    positions = {model: np.load(f'Evaluation/EvaluateModels/Results/positions_{benchmark}_{model}.npy') for model in data['model'].unique()}
    mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
    # Create a chess board of squares of size ch_size
    extent = 0, mask.shape[1], mask.shape[0], 0
    background = np.zeros_like(mask)
    background[mask == 1] = np.nan

    

    for ix, model in enumerate(data['model'].unique()):

        axes[ix].imshow(model_maps[model][RUN][-1], cmap = 'magma', vmin = 0, vmax = 1, zorder=10)
        axes[ix].imshow(background, cmap = 'gray_r', vmin = 0, vmax = 1, zorder=40, extent=extent, alpha=1, interpolation='bicubic')
        axes[ix].set_title(model)
        axes[ix].set_xticks([])
        axes[ix].set_yticks([])

        model_positions = positions[model][RUN]

        # Plot the positions
        for agent_id in range(model_positions.shape[1]):
            axes[ix].plot(model_positions[:, agent_id, 1], model_positions[ :, agent_id, 0], linestyle = '-', alpha = 0.3, marker='o', markersize=2, color=colors[agent_id], label=f'Agent {agent_id}' if ix == 0 else None, zorder=60)

        if ix == 0:
            axes[ix].legend().set_zorder(100)

    im = axes[-1].imshow(gt_maps[model][RUN], cmap = 'magma', vmin = 0, vmax = 1, zorder=10)
    axes[-1].imshow(background, cmap = 'gray_r', vmin = 0, vmax = 1, zorder=40, extent=extent, alpha=0.5, interpolation='bicubic')
    axes[-1].set_title('Ground Truth')
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    plt.show()