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

# Plot the results

sns.set_style("darkgrid")


sns.lineplot(data=df, x='step', y='total_reward', hue = 'Algorithm', style='Benchmark')

plt.show()