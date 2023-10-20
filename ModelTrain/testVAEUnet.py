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
	model_path = 'runs/TrainingUnet/VAEUnet_shekel/VAEUnet_shekel_train.pth'
elif args.benchmark == 'algae_bloom':
	model_path = r'runs\TrainingUnet\VAEUnet_algae_bloom_20231019-103929\VAEUnet_algae_bloom_train.pth'


model.load_state_dict(th.load(model_path))

# Test the model
model.eval()

ex = np.zeros_like(dataset[0][0][0])

fig, axs = plt.subplots(1, 5, figsize = (10, 10))
colormap = cc.cm['bgyw']
d0 = axs[0].imshow(ex, vmin=0, vmax=1, cmap=colormap)
d1 = axs[1].imshow(ex, vmin=0, vmax=1, cmap=colormap)
d2 = axs[2].imshow(ex, vmin=0, vmax=1, cmap=colormap)
d3 = axs[3].imshow(ex, vmin=0, vmax=1, cmap=colormap)
d4 = axs[4].imshow(ex, vmin=0, vmax=1, cmap='gray')


axs[0].set_title('Input (Model)')
axs[1].set_title('Input (Visit mask)')
axs[2].set_title('Output')
axs[3].set_title('Real')
axs[4].set_title('STD')


for i in range(len(dataset)):

	data_test = th.Tensor(dataset[i][0]).float().unsqueeze(0).to(device)
	# output = model.forward_with_prior(data_test)


	output = model.imagine(N=10, x=data_test)
	input_data = data_test[0,1,:,:].detach().cpu().numpy()
	input_data2 = data_test[0,0,:,:].detach().cpu().numpy()
	real_data = dataset[i][1] * mask

	mean = output[0].cpu().squeeze(0).detach().numpy()
	std = output[1].cpu().squeeze(0).detach().numpy()
	std = (std - np.min(std)) / (np.max(std) - np.min(std))

	d0.set_data(input_data)
	d1.set_data(input_data2)
	d2.set_data(mean)
	d3.set_data(real_data)
	d4.set_data(std)




	# Colorbar of the difference


	fig.canvas.draw()
	fig.canvas.flush_events()

	plt.pause(0.1)

	# plt.show()
