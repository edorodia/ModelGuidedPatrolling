# Test the model
import sys
sys.path.append('.')
import torch as th 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ModelTrain.dataset import DynamicDataset
from Models.unet import VAEUnet
from Environment.GroundTruths.AlgaeBloomGroundTruth import algae_colormap
import colorcet as cc
import cmasher as cmr
from tqdm import tqdm
import time

import argparse

parser = argparse.ArgumentParser(description='Test the model')

parser.add_argument('--benchmark', type=str, default='shekel', choices=['shekel', 'algae_bloom'])
parser.add_argument('--test_type', type=str,  default='test', choices=['test', 'validation'])
parser.add_argument('--N_episodes', type=int, default=50)
parser.add_argument('--name_type', type=str, choices=['Peaks_Variation', 'Peaks_Variation_long_run'])
args = parser.parse_args()

mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

#################### Folders and Parameters setting ####################

model_name = "NoDrone"

folder_model_1 = args.test_type + "_" + model_name + "_" + args.benchmark + "_Dataset_" + args.name_type			#folder where the dataset is

comb_type_model_1 = model_name + "_-_False_none"										#combination type of the model trained

trained_model_folder_model_1 = "runs-8_Completed_epoch_50_" + comb_type_model_1		#name of the folder runs if edited

VAEUnet_folder_model_1 = "VAEUnet_shekel_20240829-144816"								#internal name of variable name folder

#######################################################################

# Create the dataset
dataset_model_1 = DynamicDataset(path_trajectories = 'ModelTrain/'+ folder_model_1 + '/' + comb_type_model_1 +'/trajectories_{}_{}.npy'.format(args.benchmark, args.test_type),
						path_gts = 'ModelTrain/'+ folder_model_1 + '/' + comb_type_model_1 +'/gts_{}_{}.npy'.format(args.benchmark, args.test_type),
						transform=None)

# Load the first model
input_shape = dataset_model_1.trajectories.shape[2:]
model = VAEUnet(input_shape=input_shape, n_channels_in=2, n_channels_out=1, bilinear=False, scale=2).to(device)

model_path = trained_model_folder_model_1 + "/TrainingUnet" + "/"+ VAEUnet_folder_model_1 +"/VAEUnet_"+ args.benchmark +"_test.pth" 

model.load_state_dict(th.load(model_path, map_location=device))

# Test the first model
model.eval()

# takes the size of the first part of the item which are the trajectories 
# so the first 0 is to pick the first data item the second 0 is to pick the part of the trajectories
# the third 0 is to pick the first of the two matrices (which are equal in term of sizes) 
# the rest has to be picked by the zeros_like to generate the size of the matrix to draw
ex = np.zeros_like(dataset_model_1[0][0][0])

navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

step_mse_total = []
step_rmse_total = []
step_w_rmse_total = []


step_mse_simulation = []
step_rmse_simulation = []
step_w_rmse_simulation = []

simulation_steps = len(dataset_model_1) / args.N_episodes
print("simulation steps -> " + str(simulation_steps))
step_list_total = np.arange(0,simulation_steps)

count = 0

for i in tqdm(range(len(dataset_model_1))):

	data_test = th.Tensor(dataset_model_1[i][0]).float().unsqueeze(0).to(device) / 255.0

	start_time = time.time()
	output = model.imagine(N=10, x=data_test)
	end_time = time.time()

	input_data = data_test[0,1,:,:].detach().cpu().numpy()
	input_data2 = data_test[0,0,:,:].detach().cpu().numpy()
	real_data = dataset_model_1[i][1] * mask / 255

	mean = output[0].cpu().squeeze(0).detach().numpy()
	
	std = output[1].cpu().squeeze(0).detach().numpy()
	#rescale standard deviation between 0 and 1
	std = (std - np.min(std)) / (np.max(std) - np.min(std))

	MSE = np.sum((mean - real_data)**2)/np.sum(navigation_map)
	RMSE = np.sqrt(MSE)
	W_RMSE = np.sqrt((np.sum(((mean - real_data)**2) * real_data))/np.sum(real_data))

	step_mse_simulation.append(MSE)
	step_rmse_simulation.append(RMSE)
	step_w_rmse_simulation.append(W_RMSE)

	count += 1

	if count == simulation_steps :
		
		step_mse_total.append(step_mse_simulation)
		step_rmse_total.append(step_rmse_simulation)
		step_w_rmse_total.append(step_w_rmse_simulation)

		step_mse_simulation = []
		step_rmse_simulation = []
		step_w_rmse_simulation = []

		count = 0

#peaks_variation
#peaks_variation_long_run


np.save('step_list_'+ model_name + '_' + args.name_type + '.npy', step_list_total)
np.save('step_mse_'+ model_name + '_' + args.name_type + '.npy', np.mean(step_mse_total, axis = 0))
np.save('step_rmse_'+ model_name + '_' + args.name_type + '.npy', np.mean(step_rmse_total, axis = 0))
np.save('step_w_rmse_'+ model_name + '_' + args.name_type + '.npy', np.mean(step_w_rmse_total, axis = 0))
