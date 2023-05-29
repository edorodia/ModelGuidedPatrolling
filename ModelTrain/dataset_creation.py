import sys
sys.path.append('.')
from Models.VAE import VAE
from PathPlanners.LawnMower import LawnMowerAgent
from PathPlanners.NRRA import WanderingAgent
from Environment.PatrollingEnvironment import DiscreteModelBasedPatrolling
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch as th
import argparse

# Define the parameters of the environment

argparser = argparse.ArgumentParser()

argparser.add_argument('--n_agents', type=int, default=4)
argparser.add_argument('--frameskip', type=int, default=5)
argparser.add_argument('--max_frames', type=int, default=100)
argparser.add_argument('--N_episodes', type=int, default=5000)
argparser.add_argument('--parallel', type=bool, default=True)
argparser.add_argument('--benchmark', type=str, default='algae_bloom', choices=['algae_bloom', 'shekel'])
argparser.add_argument('--set', type=str, default='train', choices=['train', 'test'])

args = argparser.parse_args()

navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

N = args.n_agents
frameskip = args.frameskip
max_frames = args.max_frames
N_episodes = args.N_episodes
parallel = args.parallel
benchmark = args.benchmark
dataset = args.set

initial_positions = np.array([[42,32],
							[50,40],
							[43,44],
							[35,45]])


def generate_trajectory(seed):


	""" Play a game with the environment and return the trajectory of the agents and the ground truth """
	env = DiscreteModelBasedPatrolling(n_agents=N,
								navigation_map=navigation_map,
								initial_positions=initial_positions,
								model_based=True,
								movement_length=2,
								resolution=1,
								influence_radius=2,
								forgetting_factor=2,
								max_distance=200,
								benchmark='algae_bloom',
								dynamic=False,
								reward_weights=[10.0, 100.0],
								reward_type='local_changes',
								model='miopic',
								seed=seed,)


	env.reset()
	done = {i: False for i in range(N)}

	#agent = {i: LawnMowerAgent( world=map, number_of_actions=8, movement_length= 3, forward_direction=0, seed=0) for i in range(N)}
	agent = {i: WanderingAgent( world=navigation_map, number_of_actions=8, movement_length= 3, seed=seed) for i in range(N)}

	# Get the ground truth
	ground_truth = env.ground_truth.read().copy()

	# W
	W_list = []
	model_list = []

	t = 0
	
	while not all(done.values()):

		actions = {i: agent[i].move(env.fleet.vehicles[i].position.astype(int)) for i in done.keys() if not done[i]}
		_,_,done,_ = env.step(actions)

		# Get the ground truth

		if t % frameskip == 0:
			W_list.append(env.fleet.idleness_map.copy())
			model_list.append(env.model.predict().copy())

		t += 1

		if t >= max_frames:
			break

	W_list = np.asarray(W_list)
	model_list = np.asarray(model_list)

	observation_trajectory = np.stack((W_list, model_list), axis=1)


	return observation_trajectory, ground_truth


if __name__ == "__main__":

	# Create a Pool of sub-processes

	if dataset == 'train':
		seed_start = 0
		seed_end = N_episodes
	elif dataset == 'test':
		seed_start = 10000
		seed_end = 10000 + N_episodes

	if parallel:
		
		# Create a Pool of sub-processes
		pool = mp.Pool(mp.cpu_count())
		# Generate the trajectories in parallel
		trajectories = pool.map(generate_trajectory, range(seed_start, seed_end))
		# Close the pool
		pool.close()
	
	else:

		trajectories = [generate_trajectory(i) for i in tqdm(range(seed_start, seed_end))]

	gts = np.asarray([traj[1] for traj in trajectories])
	observations = np.asarray([traj[0] for traj in trajectories])

	# Save the trajectories 

	file_name = 'ModelTrain/Data/trajectories_' + benchmark + '_' + dataset + '.npy'
	np.save(file_name, observations)
	file_name = 'ModelTrain/Data/gts_' + benchmark + '_' + dataset + '.npy'
	np.save(file_name, gts)








