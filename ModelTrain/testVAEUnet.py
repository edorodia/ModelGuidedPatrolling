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
dataset = StaticDataset(path_trajectories = 'ModelTrain/Data/trajectories_{}_test_other.npy'.format(args.benchmark),
						path_gts = 'ModelTrain/Data/gts_{}_test_other.npy'.format(args.benchmark),
						transform=None)


# Load the model
input_shape = dataset.trajectories.shape[2:]
model = VAEUnet(input_shape=input_shape, n_channels_in=2, n_channels_out=1, bilinear=False, scale=2).to(device)

if args.benchmark == 'shekel':
	model_path = 'runs/TrainingUnet/VAEUnet_shekel/VAEUnet_shekel_train.pth'
elif args.benchmark == 'algae_bloom':
	model_path = 'runs\TrainingUnet\VAEUnet_algae_bloom_20230816-141157\VAEUnet_algae_bloom_test.pth'


model.load_state_dict(th.load(model_path))

# Test the model
model.eval()

ex = np.zeros_like(dataset[0][0][0])

fig, axs = plt.subplots(1, 5, figsize = (10, 10))
colormap = cc.cm['bgyw']
d0 = axs[0].imshow(ex, vmin=0, vmax=1, cmap=colormap)
d5 = axs[1].imshow(ex, vmin=0, vmax=1, cmap=colormap)
d1 = axs[2].imshow(ex, vmin=0, vmax=1, cmap=colormap)
d2 = axs[3].imshow(ex, vmin=0, vmax=1, cmap=colormap)
im = axs[4].imshow(ex, vmin=0, vmax=1, cmap='gray')

divider = make_axes_locatable(axs[4])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

axs[0].set_title('Input (Model)')
axs[1].set_title('Input (Visit mask)')
axs[2].set_title('Output')
axs[3].set_title('Real')
axs[4].set_title('Difference')


for i in range(len(dataset)):

	data_test = th.Tensor(dataset[i][0]).float().unsqueeze(0).to(device)
	# output = model.forward_with_prior(data_test)


	output = model.imagine(N=10, x=data_test).cpu().squeeze(0).detach().numpy()* mask

	input_data = data_test[0,1,:,:].detach().cpu().numpy() * mask
	input_data2 = data_test[0,0,:,:].detach().cpu().numpy()
	real_data = dataset[i][1] * mask

	d0.set_data(input_data)
	d1.set_data(output)
	d2.set_data(np.abs(real_data))
	d5.set_data(input_data2)


	# Colorbar of the difference


	fig.canvas.draw()
	fig.canvas.flush_events()

	plt.pause(0.1)

	# plt.show()
