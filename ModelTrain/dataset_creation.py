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

navigation_map = np.genfromtxt('Environment\Maps\map.txt', delimiter=' ')

N = 4

frameskip = 5

max_frames = 100

N_episodes = 5000


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

	pool = mp.Pool(10)

	# Generate the trajectories in parallel

	trajectories = pool.map(generate_trajectory, range(N_episodes))

	# trajectories = [generate_trajectory(i) for i in tqdm(range(N_episodes))]
	
	pool.close()

	gts = np.asarray([traj[1] for traj in trajectories])
	observations = np.asarray([traj[0] for traj in trajectories])

	# Save the trajectories 

	np.save('ModelTrain/Data/trajectories_static_algae_large.npy', observations)
	np.save('ModelTrain/Data/gts_static_algae_large.npy', gts)








