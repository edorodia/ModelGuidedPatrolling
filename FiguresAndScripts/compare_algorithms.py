import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

plt.switch_backend('TkAgg')

# Read ALL CSVS in folder
path = 'Evaluation/Patrolling/Results'

# Get all csv files in the folder
csv_files = []
for file in os.listdir(path):
	if file.endswith('.csv'):
		csv_files.append(file)

# Read all csv files
dataframes = []
for csv_file in csv_files:
	dataframes.append(pd.read_csv(f'{path}/{csv_file}'))

# Concatenate all dataframes
df = pd.concat(dataframes, ignore_index=True)

df = df[df['case'] != 'dynamic']

df = df[df['step'] <= 200]

# Plot the results

sns.set_style("darkgrid")

fig, ax = plt.subplots(2,1)

sns.lineplot(ax=ax[0], data=df[df['Benchmark'] == 'shekel'], x='step', y='r2', hue = 'Algorithm')
ax[0].set_title('Shekel')
ax[0].set_ylabel('R2')
ax[0].set_xlabel('')

sns.lineplot(ax=ax[1], data=df[df['Benchmark'] == 'algae_bloom'], x='step', y='r2', hue = 'Algorithm')
ax[1].set_title('Algae Bloom')
ax[1].set_ylabel('R2')


fig2, ax = plt.subplots(2,1)

sns.lineplot(ax=ax[0], data=df[df['Benchmark'] == 'shekel'], x='step', y='mean_weighted_idleness', hue='Algorithm')
ax[0].set_title('Shekel')
ax[0].set_ylabel(r'Average $\mathcal{W} \times \mathcal{I}$')
ax[0].set_xlabel('')

sns.lineplot(ax=ax[1], data=df[df['Benchmark'] == 'algae_bloom'], x='step', y='mean_weighted_idleness', hue='Algorithm')
ax[1].set_title('Algae Bloom')
ax[1].set_ylabel(r'Average $\mathcal{W} \times \mathcal{I}$')

plt.show()
