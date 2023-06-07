import sys 
sys.path.append('.')
from ModelTrain.dataset import StaticDataset
from Models.VAE import VAE, Autoencoder
import torch as th
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb
from rich.progress import track
import os

import time

import matplotlib
matplotlib.use('tkagg')


# Create the dataset
dataset = StaticDataset(path_trajectories = 'ModelTrain/Data/trajectories_static.npy', 
						path_gts = 'ModelTrain/Data/gts_static.npy',
						transform=None)

dataset_test = StaticDataset(path_trajectories = 'ModelTrain/Data/trajectories_static_test.npy', 
						path_gts = 'ModelTrain/Data/gts_static_test.npy',
						transform=None)


mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

print("Shape of the dataset: ", dataset.trajectories.shape)

# Show some samples in a 3x3 grid

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 3, figsize = (10, 10))

for i in range(3):
		j = np.random.randint(0, len(dataset))
		axs[i, 0].imshow(dataset[j][0][0,:,:], vmin=0, vmax=1)
		axs[i, 1].imshow(dataset[j][0][1,:,:], vmin=0, vmax=1)
		axs[i, 2].imshow(dataset[j][1], vmin=0, vmax=1)
		axs[i, 0].set_title('Time {}'.format(j))
		axs[i, 1].set_title('Model {}'.format(j))
		axs[i, 2].set_title('GT {}'.format(j))

plt.show()


# Create the dataloader
dataloader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 0)
dataloader_test = DataLoader(dataset_test, batch_size = 64, shuffle = True, num_workers = 0)

# Training Loop #

# Define the device
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Import the model
loss_weights = {'recon': 10, 'features': 1, 'kl': 1}
model = VAE(input_size=dataset[0][0].shape, latent_size=256, output_channels=1, loss_weights = loss_weights).to(device)

# Define the optimizer
optimizer = th.optim.Adam(model.parameters(), lr = 1e-3)
#NOTE: The loss function is defined in the model

# Define the number of epochs
N_epochs = 20

# Define the number of batches
N_batches = len(dataloader)

# Start the training loop

# Remove the previous tensorboard log and create a new one with the current time
dir_path = 'runs/TrainingVAE/VAE_Shekel_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
os.system('rm -rf ' + dir_path)
writer = tb.SummaryWriter(log_dir=dir_path, comment='VAE_static_training_shekel')


for epoch in track(range(N_epochs), description="Training progress: "):

	running_loss = []
	running_recon_loss = []
	running_features_loss = []
	running_kl_loss = []

	model.train()
	for i, data in enumerate(dataloader):

		# Get the batch
		batch, batch_gt = data

		# Transform the batch to a float Tensor for the model
		batch = th.Tensor(batch).float().to(device)
		batch_gt = th.Tensor(batch_gt).float().to(device)
		batch_gt = batch_gt.unsqueeze(1)

		# Forward pass
		output, mu, logvar = model(batch)

		# Compute the loss
		loss, recon_loss, kl_div, features_loss = model.loss(x=output, x_hat=batch_gt, mask=mask, mu=mu, logvar=logvar)

		# Add the loss to the running loss
		running_loss.append(loss.item())
		running_recon_loss.append(recon_loss.item())
		running_features_loss.append(features_loss.item())
		running_kl_loss.append(kl_div.item())

		# Backward pass
		optimizer.zero_grad()
		
		loss.backward()
		th.nn.utils.clip_grad_norm_(model.parameters(), 10)
		optimizer.step()

	# Test the model
	model.eval()
	with th.no_grad():

		running_test_loss = []
		running_test_recon_loss = []
		running_test_features_loss = []
		running_test_kl_loss = []

		# Get the batch

		for i, data_test in enumerate(dataloader_test):

			# Get the batch
			batch, batch_gt = data_test
			# Transform the batch to a float Tensor for the model
			batch = th.Tensor(batch).float().to(device)
			batch_gt = th.Tensor(batch_gt).float().to(device)
			batch_gt = batch_gt.unsqueeze(1)

			# Forward pass
			output, mu, logvar = model(batch)
			# Compute the loss
			test_loss, test_recon_loss, test_kl_div, test_features_loss = model.loss(x=output, x_hat=batch_gt, mask=mask, mu=mu, logvar=logvar)
			# Add the loss to the running loss
			running_test_loss.append(test_loss.item())
			running_test_recon_loss.append(test_recon_loss.item())
			running_test_features_loss.append(test_features_loss.item())
			running_test_kl_loss.append(test_kl_div.item())

		
	# Save the model if the loss is lower than the previous one
	if epoch == 0:
		th.save(model.state_dict(), dir_path + '/VAE_static_shekel.pth')
		min_loss = np.mean(running_test_loss)
	elif np.mean(running_test_loss) < min_loss:
		th.save(model.state_dict(), dir_path + '/VAE_static_shekel_{}.pth'.format(epoch))
		print("Model saved at epoch {}".format(epoch))
		min_loss = np.mean(running_test_loss)
	


	# Add the test loss to the tb writer
	writer.add_scalar('Test/Loss', np.mean(running_test_loss), epoch)
	writer.add_scalar('Test/ReconLoss', np.mean(running_test_recon_loss), epoch)
	writer.add_scalar('Test/FeaturesLoss', np.mean(running_test_features_loss), epoch)
	writer.add_scalar('Test/KLdiv', np.mean(running_test_kl_loss), epoch)


	# Add the loss to the tb writer
	writer.add_scalar('Train/Loss', np.mean(running_loss), epoch)
	writer.add_scalar('Train/ReconLoss', np.mean(running_recon_loss), epoch)
	writer.add_scalar('Train/FeaturesLoss', np.mean(running_features_loss), epoch)
	writer.add_scalar('Train/KLdiv', np.mean(running_kl_loss), epoch)

	# Print the loss
	print("Epoch: {}/{} Total Loss: {} Recon Loss: {} Features Loss: {} KL Div Loss {}".format(epoch, N_epochs, np.mean(running_test_loss), np.mean(running_test_recon_loss), np.mean(running_test_features_loss), np.mean(running_test_kl_loss)))





