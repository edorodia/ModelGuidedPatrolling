# Test the model
import sys
sys.path.append('.')
import torch as th 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ModelTrain.dataset import StaticDataset
from Models.unet import VAEUnet
from Environment.GroundTruths.AlgaeBloomGroundTruth import algae_colormap
import colorcet as cc
import cmasher as cmr
from tqdm import tqdm
import time

import argparse

parser = argparse.ArgumentParser(description='Test the model')

parser.add_argument('--benchmark', type=str, default='shekel', choices=['shekel', 'algae_bloom'])
parser.add_argument('--test_type', type=str,  choices=['test', 'validation'])
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

#################### Folders and Parameters setting ####################

folder = args.test_type + "_NoDrone_" + args.benchmark + "_Datasets"			#folder where the dataset is

comb_type = "NoDrone_-_False_none"										#combination type of the model trained

trained_model_folder = "runs-8_Completed_epoch_50_" + comb_type		#name of the folder runs if edited

VAEUnet_folder = "VAEUnet_shekel_20240829-144816"								#internal name of variable name folder

#######################################################################

# Create the dataset
dataset = StaticDataset(path_trajectories = 'ModelTrain/'+ folder + '/' + comb_type +'/trajectories_{}_{}.npy'.format(args.benchmark, args.test_type),
						path_gts = 'ModelTrain/'+ folder + '/' + comb_type +'/gts_{}_{}.npy'.format(args.benchmark, args.test_type),
						transform=None)


# Load the model
input_shape = dataset.trajectories.shape[2:]
model = VAEUnet(input_shape=input_shape, n_channels_in=2, n_channels_out=1, bilinear=False, scale=2).to(device)

"""
if args.benchmark == 'shekel':
	model_path = r'runs\optuna\shekel\VAEUnet_shekel_test_trial_num_18.pth'
	model_path = trained_model_folder + "/TrainingUnet"
elif args.benchmark == 'algae_bloom':
	model_path = r'runs\optuna\algae_bloom\VAEUnet_algae_bloom_test_trial_num_12.pth'
"""

model_path = trained_model_folder + "/TrainingUnet" + "/"+ VAEUnet_folder +"/VAEUnet_"+ args.benchmark +"_test.pth" 

model.load_state_dict(th.load(model_path, map_location=device))

# Test the model
model.eval()

# takes the size of the first part of the item which are the trajectories 
# so the first 0 is to pick the first data item the second 0 is to pick the part of the trajectories
# the third 0 is to pick the first of the two matrices (which are equal in term of sizes) 
# the rest has to be picked by the zeros_like to generate the size of the matrix to draw
ex = np.zeros_like(dataset[0][0][0])

#print(ex.shape)


if args.render:
	cmap = cmr.get_sub_cmap('cmr.toxic', 0.30, 0.99)

	fig, axs = plt.subplots(1, 5, figsize = (10, 10))

	axs[1].imshow(mask, vmin=0, vmax=1, cmap = 'copper_r', alpha = 1 - mask, zorder=10)
	axs[2].imshow(mask, vmin=0, vmax=1, cmap = 'copper_r', alpha = 1 - mask, zorder=10)
	axs[3].imshow(mask, vmin=0, vmax=1, cmap = 'copper_r', alpha = 1 - mask, zorder=10)
	axs[4].imshow(mask, vmin=0, vmax=1, cmap = 'copper_r', alpha = 1 - mask, zorder=10)

	d0 = axs[0].imshow(ex, vmin=0, vmax=1, cmap=cmap)
	d1 = axs[1].imshow(ex, vmin=0, vmax=1, cmap=cmap)
	d2 = axs[2].imshow(ex, vmin=0, vmax=1, cmap=cmap)
	d3 = axs[3].imshow(ex, vmin=0, vmax=1, cmap=cmap)
	d4 = axs[4].imshow(ex, vmin=0, vmax=1, cmap='gray')


	axs[0].set_title('Input (Model)')
	axs[1].set_title('Input (Visit mask)')
	axs[2].set_title('Output')
	axs[3].set_title('Real')
	axs[4].set_title('STD')

navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

SUM_MSE = 0
SUM_R2_SCORE = 0
SUM_RMSE = 0
SUM_W_RMSE = 0
SUM_ELAPSED_TIME_SECONDS = 0

for i in tqdm(range(len(dataset))):

	data_test = th.Tensor(dataset[i][0]).float().unsqueeze(0).to(device) / 255.0 
	#output = model.forward_with_prior(data_test)

	start_time = time.time()
	output = model.imagine(N=10, x=data_test)
	end_time = time.time()

	input_data = data_test[0,1,:,:].detach().cpu().numpy()
	input_data2 = data_test[0,0,:,:].detach().cpu().numpy()
	real_data = dataset[i][1] * mask / 255

	mean = output[0].cpu().squeeze(0).detach().numpy()

	#mean = mask*(mean - np.min(mean)) / (np.max(mean) - np.min(mean))
	#mean = input_data2*input_data + (1-input_data2)*mean
	std = output[1].cpu().squeeze(0).detach().numpy()
	#rescale standard deviation between 0 and 1
	std = (std - np.min(std)) / (np.max(std) - np.min(std))


	if args.render:
		d0.set_data(input_data)
		d1.set_data(input_data2)
		d2.set_data(mean)
		d3.set_data(real_data)
		d4.set_data(std)


	MSE = np.sum((mean - real_data)**2)/np.sum(navigation_map)
	R2_SCORE = 1 - np.sum((real_data - mean) ** 2) / np.sum((real_data - np.mean(real_data)) ** 2)
	RMSE = np.sqrt(MSE)
	W_RMSE = np.sqrt((np.sum(((mean - real_data)**2) * real_data))/np.sum(real_data))
	ELAPSED_TIME_SECONDS = end_time - start_time

	SUM_MSE += MSE
	SUM_R2_SCORE += R2_SCORE
	SUM_RMSE += RMSE
	SUM_W_RMSE += W_RMSE
	SUM_ELAPSED_TIME_SECONDS += ELAPSED_TIME_SECONDS
	#print("ERROR for " + str(i) + " data_step: " + str(MSE))


	# Colorbar of the difference

	if args.render:
		fig.canvas.draw()
		fig.canvas.flush_events()

		plt.pause(0.1)

	# plt.show()

AVG_MSE = SUM_MSE / len(dataset)
AVG_R2_SCORE = SUM_R2_SCORE / len(dataset)
AVG_RMSE = SUM_RMSE / len(dataset)
AVG_W_RMSE = SUM_W_RMSE / len(dataset)
AVG_ELAPSED_TIME_SECONDS = SUM_ELAPSED_TIME_SECONDS / len(dataset)

with open(args.test_type + '_' + args.benchmark + '_' + comb_type + '.txt', 'w') as file:
	file.write(f"{'MSE':<15} -: {AVG_MSE:.20f}\n")
	file.write(f"{'RMSE':<15} -: {AVG_RMSE:.20f}\n")
	file.write(f"{'W_RMSE':<15} -: {AVG_W_RMSE:.20f}\n")
	file.write(f"{'SEC_PER_OP':<15} -: {AVG_ELAPSED_TIME_SECONDS:.20f}\n")
	file.write(f"{'R2':<15} -: {AVG_R2_SCORE:.20f}\n")
	
	
print(AVG_MSE)
print(AVG_RMSE)
print(AVG_W_RMSE)
print(AVG_ELAPSED_TIME_SECONDS)
print(AVG_R2_SCORE)

