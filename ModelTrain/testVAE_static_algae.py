# Test the model
import sys
sys.path.append('.')
import torch as th 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('tkagg')

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ModelTrain.dataset import StaticDataset
from Models.VAE import VAE
from Models.SegNet import SegNet

from Environment.GroundTruths.AlgaeBloomGroundTruth import algae_colormap

mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Create the dataset
dataset = StaticDataset(path_trajectories = 'ModelTrain/Data/trajectories_static_algae_test.npy',

						path_gts = 'ModelTrain/Data/gts_static_algae_test.npy',

						transform=None)


# Load the model
#model = VAE(input_size=dataset[0][0].shape, latent_size=256, output_channels=1, loss_weights = None).to(device)
model = SegNet(input_size=dataset[0][0].shape, latent_size=256, output_channels=1, loss_weights = None).to(device)
#model.load_state_dict(th.load('runs\TrainingVAE\VAE_Algae_20230525-224602\VAE_static_algae_8.pth'))
model.load_state_dict(th.load('runs\TrainingUnet\VAEUnet_algae_bloom_20230816-131702\VAEUnet_algae_bloom_test.pth'))

# Test the model
model.eval()

ex = np.zeros_like(dataset[0][0][0])

fig, axs = plt.subplots(1, 4, figsize = (10, 10))
d0 = axs[0].imshow(ex, vmin=0, vmax=1, cmap='magma')
d1 = axs[1].imshow(ex, vmin=0, vmax=1, cmap='magma')
d2 = axs[2].imshow(ex, vmin=0, vmax=1, cmap='magma')
im = axs[3].imshow(ex, vmin=0, vmax=1, cmap='gray')

divider = make_axes_locatable(axs[3])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

axs[0].set_title('Input')
axs[1].set_title('Output')
axs[2].set_title('Real')
axs[3].set_title('Difference')


for i in range(len(dataset)):

	data_test = th.Tensor(dataset[i][0]).float().unsqueeze(0).to(device)
	output,_,_ = model(data_test)

	input_data = data_test[0,1,:,:].detach().cpu().numpy() * mask
	output_data = output[0,0,:,:].detach().cpu().numpy() * mask
	real_data = dataset[i][1] * mask

	d0.set_data(input_data)
	d1.set_data(output_data)
	d2.set_data(np.abs(real_data))
	im.set_data(np.abs(output_data - real_data))


	# Colorbar of the difference


	fig.canvas.draw()
	fig.canvas.flush_events()

	plt.pause(0.01)

	# plt.show()
