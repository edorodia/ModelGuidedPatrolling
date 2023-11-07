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

import argparse

parser = argparse.ArgumentParser(description='Test the model')

parser.add_argument('--benchmark', type=str, default='algae_bloom', choices=['shekel', 'algae_bloom'])
args = parser.parse_args()

mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Create the dataset
dataset = StaticDataset(path_trajectories = 'ModelTrain/Data/trajectories_{}_test.npy'.format(args.benchmark),
						path_gts = 'ModelTrain/Data/gts_{}_test.npy'.format(args.benchmark),
						transform=None)


# Load the model
input_shape = dataset.trajectories.shape[2:]
model = VAEUnet(input_shape=input_shape, n_channels_in=2, n_channels_out=1, bilinear=False, scale=2).to(device)

if args.benchmark == 'shekel':
	model_path = r'runs\optuna\shekel\VAEUnet_shekel_test_trial_num_18.pth'
elif args.benchmark == 'algae_bloom':
	model_path = r'runs\optuna\algae_bloom\VAEUnet_algae_bloom_test_trial_num_12.pth'


model.load_state_dict(th.load(model_path, map_location=device))

# Test the model
model.eval()

ex = np.zeros_like(dataset[0][0][0])

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

for i in range(len(dataset)):

	data_test = th.Tensor(dataset[i][0]).float().unsqueeze(0).to(device) / 255.0 
	#output = model.forward_with_prior(data_test)


	output = model.imagine(N=10, x=data_test)
	input_data = data_test[0,1,:,:].detach().cpu().numpy()
	input_data2 = data_test[0,0,:,:].detach().cpu().numpy()
	real_data = dataset[i][1] * mask / 255

	mean = output[0].cpu().squeeze(0).detach().numpy()

	#mean = mask*(mean - np.min(mean)) / (np.max(mean) - np.min(mean))
	#mean = input_data2*input_data + (1-input_data2)*mean
	std = output[1].cpu().squeeze(0).detach().numpy()
	std = (std - np.min(std)) / (np.max(std) - np.min(std))

	d0.set_data(input_data)
	d1.set_data(input_data2)
	d2.set_data(mean)
	d3.set_data(real_data)
	d4.set_data(std)


	print("ERROR: ", np.sum(np.abs(mean - real_data)**2)/np.sum(navigation_map))


	# Colorbar of the difference


	fig.canvas.draw()
	fig.canvas.flush_events()

	plt.pause(0.1)

	# plt.show()
