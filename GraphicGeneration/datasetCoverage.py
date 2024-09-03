# Check after how many average steps the lake is covered by a percentage indicated as parameter
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
parser.add_argument('--threshold', type=float, default=90)
args = parser.parse_args()

print(args.threshold)

mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

#################### Folders and Parameters setting ####################

folder = args.test_type + "_NoNoise_" + args.benchmark + "_Datasets"			#folder where the dataset is

#to avoid noise problems we use the True_False_none combination in order to use the covered positions part of the trajectories
comb_type = "NoNoise_-_True_False_none"										#combination type of the model trained

#######################################################################


data = np.load('ModelTrain/'+ folder + '/' + comb_type +'/trajectories_{}_{}.npy'.format(args.benchmark, args.test_type))

#data = np.load('ModelTrain/Data' + '/trajectories_{}_{}.npy'.format(args.benchmark, args.test_type))

ex = np.zeros_like(data[0][0][0])

cmap = cmr.get_sub_cmap('cmr.toxic', 0.30, 0.99)

fig, axs = plt.subplots(1, 2, figsize = (10, 10))

d0 = axs[0].imshow(ex, vmin=0, vmax=1, cmap=cmap)
d1 = axs[1].imshow(ex, vmin=0, vmax=1, cmap=cmr.get_sub_cmap('cmr.toxic', 0.30, 0.99))

axs[0].set_title('Visited')
axs[1].set_title('Importance')

navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

sum_importance_coverage_step = 0
added = False
found = False


for i in tqdm(range(len(data))):

	#data_test = th.Tensor(data[i][0]).float().unsqueeze(0).to(device) / 255.0 
	
	input_data2 = data[i,:,:,:,:]

	for j in range(len(input_data2)):


		if args.render :
			d0.set_data(input_data2[j,1,:,:])

		importance = input_data2[j,1,:,:] * navigation_map
		
		visited = input_data2[j,0,:,:] * navigation_map

		#print(str(len(importance[importance > 0])) + "-" + str(len(visited[visited > 0])))

		#here we can check how much coverage has the importance on the whole surface of the lake
		importance_coverage = len(visited[visited > 0]) / np.sum(navigation_map)
		importance_coverage = importance_coverage * 100

		#print(str(len(input_data[input_data > 0]) / np.sum(navigation_map)) + "-" + str(len(input_data1[input_data1 > 0]) / np.sum(navigation_map)))
		
		if importance_coverage >= args.threshold :
			sum_importance_coverage_step += j
			added = True
			break

		if args.render :
			fig.canvas.draw()
			fig.canvas.flush_events()

			plt.pause(0.1)

	if added == False :
		sum_importance_coverage_step += len(input_data2)
	else:
		added = False

print(sum_importance_coverage_step / len(data))


