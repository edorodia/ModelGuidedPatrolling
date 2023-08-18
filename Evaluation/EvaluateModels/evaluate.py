import sys
sys.path.append('.')
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
import argparse
import time
from tqdm.auto import tqdm
from PathPlanners.NRRA import WanderingAgent

seed = 50000
dynamic = True
all_models = ['knn', 'miopic', 'gp', 'deepUnet', 'vaeUnet']
# all_models = ['vaeUnet']
all_benchmarks = ['algae_bloom', 'shekel']
all_benchmarks = [ 'shekel']

if dynamic:
	output_path = 'Evaluation/EvaluateModels/Results/results_all_dynamic.csv'
else:
	output_path = 'Evaluation/EvaluateModels/Results/results_all.csv'

# Parse the arguments
parser = argparse.ArgumentParser(description='Evaluate the models')
parser.add_argument('--model', type=str, default='all', help='Model to evaluate', choices=all_models)
parser.add_argument('--benchmark', type=str, default='all', help='Benchmark to evaluate', choices = ['all', 'algae_bloom', 'shekel'])
parser.add_argument('--runs', type=int, default=50, help='Number of runs')
parser.add_argument('--N', type=int, default=4, help='Number of agents')
parser.add_argument('--max_frames', type=int, default=100, help='Maximum number of frames')
parser.add_argument('--render', type=bool, default=False, help='Render mode')
parser.add_argument('--save_maps', type=bool, default=False, help='Save maps')
parser.add_argument('--append', type=bool, default=True, help='Append to the csv file')

args = parser.parse_args()


RUNS = args.runs
N = args.N
initial_positions = np.array([[42,32],
							[50,40],
							[43,44],
							[35,45]])

models = all_models if args.model == 'all' else [args.model]
navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
max_frames = args.max_frames
benchmarks = all_benchmarks if args.benchmark == 'all' else [args.benchmark]


metrics = []

for benchmark in tqdm(benchmarks, desc='Benchmark: ', total=len(benchmarks), leave=False, position=0):

	for model in tqdm(models, desc='Model: ', total=len(models), leave=False):

		# Create the environment

		env = DiscreteModelBasedPatrolling(n_agents=N,
								   navigation_map=navigation_map,
								   initial_positions=initial_positions,
								   model_based=True,
								   movement_length=2,
								   resolution=1,
								   influence_radius=1.5 if benchmark == 'algae_bloom' else 2,
								   forgetting_factor=2,
								   max_distance=200,
								   benchmark=benchmark,
								   dynamic=dynamic,
								   reward_weights=[10.0, 100.0],
								   reward_type='local_changes',
								   model=model,
								   seed=seed,)

		agent = {i: WanderingAgent( world=navigation_map, number_of_actions=8, movement_length= 3, seed=10000) for i in range(N)}
		
		all_positions = []
		gt_map = []
		all_models = []
		
		for run in tqdm(range(RUNS), desc='Run: ', total=RUNS, leave=False):

			# Reset the environment
			env.reset()
			model_map = []
			gt_map.append(env.ground_truth.read().copy())

			done = {i: False for i in range(N)}

			positions = []
			positions.append(env.fleet.get_positions().copy())

			for step in tqdm(range(max_frames), desc='Step: ', total=max_frames, leave=False):

	
				t0 = time.time()
				actions = {i: agent[i].move(env.fleet.vehicles[i].position.astype(int)) for i in done.keys() if not done[i]}
				observations, rewards, done, info = env.step(actions)

				# Append a new row to the dataframe
				metrics.append([run, step, benchmark, model, info['mse'], info['rmse'], info['weighted_rmse'], info['R2'], time.time()-t0])

				if args.render:
					env.render()

				if args.save_maps and step % 10 == 0:
					model_map.append(env.model.predict().copy())
				
				positions.append(env.fleet.get_positions().copy())

			all_positions.append(np.array(positions).copy())
			all_models.append(np.array(model_map).copy())

		# Save gt maps
		if args.save_maps:
			if not dynamic:
				np.save(f'Evaluation/EvaluateModels/Results/positions_{benchmark}_{model}.npy', np.array(all_positions))
				np.save(f'Evaluation/EvaluateModels/Results/gts_evaluation_{benchmark}_{model}.npy', np.array(gt_map))
				np.save(f'Evaluation/EvaluateModels/Results/estimated_model_{benchmark}_{model}.npy', np.array(all_models))
			else:
				np.save(f'Evaluation/EvaluateModels/Results/positions_{benchmark}_{model}_dynamic.npy', np.array(all_positions))
				np.save(f'Evaluation/EvaluateModels/Results/gts_evaluation_{benchmark}_{model}_dynamic.npy', np.array(gt_map))
				np.save(f'Evaluation/EvaluateModels/Results/estimated_model_{benchmark}_{model}_dynamic.npy', np.array(all_models))


if args.save_maps == False:
	if args.append:
		# Load the metrics
		old_metrics = pd.read_csv(output_path)
		# Delete the rows with the same benchmark as benchmarks and model as models
		old_metrics = old_metrics[~((old_metrics['benchmark'].isin(benchmarks)) & (old_metrics['model'].isin(models)))]
		# Concatenate the new metrics
		new_metrics = pd.concat((old_metrics, pd.DataFrame(metrics, columns=['run', 'step', 'benchmark', 'model', 'mse', 'rmse', 'weighted_rmse', 'R2', 'time'])))
		# Save the metrics
		new_metrics.to_csv(output_path, index=False)
	else:
		metrics = pd.DataFrame(metrics, columns=['run', 'step', 'benchmark', 'model', 'mse', 'rmse', 'weighted_rmse', 'R2', 'time'])
		metrics.to_csv(output_path, index=False)