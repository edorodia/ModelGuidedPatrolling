from npy_append_array import NpyAppendArray
import sys
sys.path.append('.')
from Models.VAE import VAE
from PathPlanners.LawnMower import LawnMowerAgent
from PathPlanners.NRRA import WanderingAgent
from Environment.PatrollingEnvironment import DiscreteModelBasedHetPatrolling
from Environment.TimedDiscreteModelBasedHetPatrolling import TimedDiscreteModelBasedHetPatrolling
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch as th
import argparse
from HetPathPlanners.RandomMover import RandomDroneMover, RandomVehicleMover

# Define the parameters of the environment

argparser = argparse.ArgumentParser()

# argparser.add_argument('--n_agents', type=int, default=4)
# argparser.add_argument('--frameskip', type=int, default=20)
# argparser.add_argument('--max_frames', type=int, default=100)
# argparser.add_argument('--N_episodes', type=int, default=5000)
# argparser.add_argument('--parallel', type=bool, default=False)
# argparser.add_argument('--benchmark', type=str, default='algae_bloom', choices=['algae_bloom', 'shekel'])
# argparser.add_argument('--set', type=str, default='train', choices=['train', 'test'])
# argparser.add_argument('--random', type=bool, default=False)

argparser.add_argument('--n_agents', type=int, default=4)
argparser.add_argument('--n_drones', type=int, default=1)
argparser.add_argument('--frameskip', type=int, default=1)
argparser.add_argument('--max_frames', type=int, default=319)
argparser.add_argument('--N_episodes', type=int, default=3000)
#if the argument is a string, even "false" then bool('False') returns True giving parallel a True value
argparser.add_argument('--parallel', action='store_true')
argparser.add_argument('--benchmark', type=str, default='shekel', choices=['algae_bloom', 'shekel'])
argparser.add_argument('--set', type=str, default='train', choices=['train', 'test'])
argparser.add_argument('--random', type=bool, default=False)

###
"""	ADD THE PARAMETERS FOR THE NOISE MODEL and SPEED RATIO
	this two parameters have influence on the visited map and the importance read 
	in the noise models there is also the fisheye one where we can choose the width of the central square where data stay true to ground truth
	this can be parametrized"""
###
argparser.add_argument('--drone_noise', type=str, default='NoNoise', choices=['FishEyeNoise', 'MeanNoise', 'NoNoise'])
argparser.add_argument('--no_noise_side', type = int, default=None)
argparser.add_argument('--speed_ratio', type=float, default=11.67)
argparser.add_argument('--influence_drone_visited_map', action='store_true')
argparser.add_argument('--influence_asv_visited_map', action='store_true')
argparser.add_argument('--importance_asv_read', type=str, default='miopic', choices=['miopic', 'none'])

args = argparser.parse_args()

if args.no_noise_side is None:
	if args.drone_noise == 'FishEyeNoise':
		argparser.error("When --drone_noise id 'FishEyeNoise' --no_noise_side must be set")

navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

N_ASV = args.n_agents
N_DRONES = args.n_drones
frameskip = args.frameskip
max_frames = args.max_frames
N_episodes = args.N_episodes
parallel = args.parallel
benchmark = args.benchmark
dataset = args.set
drone_noise = args.drone_noise
no_noise_side = args.no_noise_side
speed_ratio = args.speed_ratio
influence_drone_visited_map = args.influence_drone_visited_map
influence_asv_visited_map = args.influence_asv_visited_map
importance_asv_read = args.importance_asv_read

initial_ASV_positions = np.array([[42, 32],
									  [50, 40],
									  [43, 44],
									  [35, 45]])
		
initial_drone_position = np.array([[16,24]])

def generate_trajectory(seed):

	if parallel:
		np.random.seed(seed)

	""" Play a game with the environment and return the trajectory of the agents and the ground truth """
	patrollingModel = DiscreteModelBasedHetPatrolling( initial_air_positions = initial_drone_position,
					max_air_distance = 1000,
					influence_side = 9,
					forgetting_air_factor = 0.01,	
					drone_idleness_influence = 0.20,
					n_agents = N_ASV,
					navigation_map = navigation_map,
					initial_positions = initial_ASV_positions,
					model_based = True,
					movement_length = 2,
					resolution = 1,
					max_distance = 400,
					influence_radius = 2,
					forgetting_factor= 0.01,
					reward_drone_type='weighted_idleness',
					reward_type='weighted_idleness',
					reward_weights=[10, 10],
					benchmark = args.benchmark,
					model = importance_asv_read,	#set the model from parameter in order to have radius importance read activated or not
					dynamic = False,
					seed = seed,
					int_observation = True,
					previous_exploration = False,
					pre_exploration_policy = None,
					pre_exploration_steps = 0, 
					camera_fov_angle = 160,						
					drone_height = 120,							
					n_drones = N_DRONES,									
					drone_direct_idleness_influece = False,		
					blur_data = False,
					drone_noise = drone_noise,
					fisheye_side = no_noise_side,
					update_only_with_ASV = False,
					influence_drone_visited_map = influence_drone_visited_map,
					influence_asv_visited_map = influence_asv_visited_map,
					check_max_distances = False
					)
	
	timedEnv = TimedDiscreteModelBasedHetPatrolling(	env = patrollingModel,
												speed_ratio = speed_ratio,
												asv_path_planner = RandomVehicleMover,
												drone_path_planner = RandomDroneMover,
												no_render = True,
												no_print = True)

	
	

	# Get the ground truth
	ground_truth = timedEnv.env.ground_truth.read().copy()

	# W
	W_list = []
	model_list = []

	t = 0


	frame_number = np.random.choice(np.arange(0, max_frames), size = max_frames//frameskip, replace=False)
	
	while t < max_frames + 1:
		
		timedEnv.step()

		# Get the ground truth

		# Extracts the frames from the random range from 0 to max_Frames calculated earlier
		if t in frame_number and args.random:
			ASV_visited_map = timedEnv.env.fleet.visited_map.copy()
			Drone_visited_map = timedEnv.env.fleet.visited_air_map.copy()
			W_list.append(np.logical_or(ASV_visited_map, Drone_visited_map))
			#W_list.append(timedEnv.env.fleet.visited_map.copy())
			model_list.append(timedEnv.env.model.predict().copy())
		#in this case there isn't the random flag active, this means that regularly, every frameskip, a frame is extracted
		elif t % frameskip == 0 and not args.random:
			ASV_visited_map = timedEnv.env.fleet.visited_map.copy()
			Drone_visited_map = timedEnv.env.fleet.visited_air_map.copy()
			W_list.append(np.logical_or(ASV_visited_map, Drone_visited_map))
			#W_list.append(timedEnv.env.fleet.visited_map.copy())
			model_list.append(timedEnv.env.model.predict().copy())


		t += 1

		if t >= max_frames + 1:
			break

	W_list = np.asarray(W_list)
	model_list = np.asarray(model_list)

	observation_trajectory = np.stack((W_list, model_list), axis=1)
	
	# print("Hi! I'm process {} and I'm done!".format(seed))

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
		pool = mp.Pool(2)
		# Generate the trajectories in parallel imap returns an iterator
		#Careful imap should return an iterable that you need to elaborate to extract useful information about trajectories
		trajectories = list(pool.imap(generate_trajectory, range(seed_start, seed_end)))
		# Close the pool
		pool.close()
	
	else:

		file_name_traj = 'ModelTrain/Data/trajectories_' + benchmark + '_' + dataset + '.npy'
		file_name_gts = 'ModelTrain/Data/gts_' + benchmark + '_' + dataset + '.npy'
		
		with NpyAppendArray(file_name_gts, delete_if_exists=True) as fnGTS, NpyAppendArray(file_name_traj, delete_if_exists=True) as fnTRAJ:
			for i in tqdm(range(seed_start, seed_end)):
				trajectorie = generate_trajectory(i)
				gts = np.asarray([trajectorie[1]])
				observations = np.asarray([trajectorie[0]])
				fnGTS.append((gts* 255.0).astype(np.uint8))
				fnTRAJ.append((observations * 255.0).astype(np.uint8))
		

		#trajectories = [generate_trajectory(i) for i in tqdm(range(seed_start, seed_end))]

	"""for obs, gt in trajectories:
		gts = np.asarray(gt)
		observations = np.asarray(obs)"""

	#gts = np.asarray([traj[1] for traj in trajectories])
	#observations = np.asarray([traj[0] for traj in trajectories])

	# Save the trajectories 
	"""
	file_name = 'ModelTrain/Data/trajectories_' + benchmark + '_' + dataset + '.npy'
	np.save(file_name, (observations * 255.0).astype(np.uint8))
	file_name = 'ModelTrain/Data/gts_' + benchmark + '_' + dataset + '.npy'
	np.save(file_name, (gts* 255.0).astype(np.uint8))
	"""








