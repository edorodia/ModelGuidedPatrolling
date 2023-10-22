import sys 
sys.path.append('.')
from ModelTrain.dataset import StaticDataset
from Models.unet import VAEUnet
import torch as th
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb
from tqdm.auto import tqdm
import os
import time
import argparse

# Define the parameters of the environment
argparser = argparse.ArgumentParser()

argparser.add_argument('--benchmark', type=str, default='algae_bloom', choices=['algae_bloom', 'shekel'])
argparser.add_argument('--epochs', type=int, default=30)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--lr', type=float, default=7e-4)
argparser.add_argument('--weight_decay', type=float, default=0)
argparser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cuda:1', 'cpu']) 
argparser.add_argument('--scale', type=int, default=2) 


args = argparser.parse_args()

benchmark = args.benchmark
train_traj_file_name = 'ModelTrain/Data/trajectories_' + benchmark + '_train.npy'
train_gt_file_name = 'ModelTrain/Data/gts_' + benchmark + '_train.npy'
test_traj_file_name = 'ModelTrain/Data/trajectories_' + benchmark + '_train.npy'
test_gt_file_name = 'ModelTrain/Data/gts_' + benchmark + '_train.npy'

# Create the dataset
dataset = StaticDataset(path_trajectories = train_traj_file_name, 
						path_gts = train_gt_file_name,
						transform=None)

dataset_test = StaticDataset(path_trajectories = test_traj_file_name, 
						path_gts = test_gt_file_name,
						transform=None)


mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

print("Shape of the dataset: ", dataset.trajectories.shape)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0, )
dataloader_test = DataLoader(dataset_test, batch_size = args.batch_size, shuffle = True, num_workers = 0)

# Training Loop #

# Define the device
device_str = args.device
device = th.device(device_str)

# Import the model
input_shape = dataset.trajectories.shape[2:]
model = VAEUnet(input_shape=input_shape, n_channels_in=2, n_channels_out=1, bilinear=False, scale=args.scale).to(device)

# Define the optimizer
optimizer = th.optim.Adam(model.parameters(), lr = args.lr)
#NOTE: The loss function is defined in the model

# Define the number of epochs
N_epochs = args.epochs

# Start the training loop

# Remove the previous tensorboard log and create a new one with the current time
dir_path = 'runs/TrainingUnet/VAEUnet_{}_{}'.format(benchmark, time.strftime("%Y%m%d-%H%M%S"))
os.system('rm -rf ' + dir_path)
writer = tb.SummaryWriter(log_dir=dir_path, comment='VAEUnet_training_{}'.format(benchmark))


def beta_scheduler(t, beta_min = 0.001, beta_max=1.0, t_beta_min=0.0, t_beta_max=0.2):

	if t < t_beta_min:
		return beta_min
	elif t_beta_min > t > t_beta_max:
		return beta_min + (beta_max - beta_min) * (t - t_beta_min) / (t_beta_max - t_beta_min)
	else:
		return beta_max

error_mask = np.genfromtxt('Environment\Maps\map.txt', delimiter=' ')
error_mask = th.Tensor(error_mask).to(device)


for epoch in tqdm(range(N_epochs), desc="Epochs: "):

	running_loss = []
	running_loss_mse = []
	running_loss_kl = []
	running_loss_perceptual = []
	model.train()

	for i, data in tqdm(enumerate(dataloader), desc="Batches: ", total= len(dataset) // args.batch_size + 1):

		# Get the batch
		batch, batch_gt = data

		# Transform the batch to a float Tensor for the model
		with th.no_grad():
			batch = th.Tensor(batch).float().to(device) / 255.0
			batch_gt = th.Tensor(batch_gt).float().to(device) / 255.0
			batch_gt = batch_gt.unsqueeze(1)

		# Forward pass
		output, prior, posterior = model(batch, batch_gt)

		# Compute the loss
		loss, recon_loss, kl_loss, perceptual_loss = model.compute_loss(x = batch[:,1,:,:], 
													x_true = batch_gt, 
													x_out = output, 
													prior = prior, 
													posterior = posterior, 
													beta = 4.565*beta_scheduler(epoch/N_epochs), 
													alpha=6.68,
													gamma=7.72,
													error_mask=error_mask)

		# Add the loss to the running loss
		running_loss.append(loss.item())
		running_loss_mse.append(recon_loss.item())
		running_loss_kl.append(kl_loss.item())
		running_loss_perceptual.append(perceptual_loss.item())
		
		# Reset the gradients
		optimizer.zero_grad()
		# Backward pass
		loss.backward()
		# Apply the gradients
		optimizer.step()

	# Test the model
	model.eval()

	with th.no_grad():

		running_test_loss = []

		# Get the batch

		test_loss = 0

		for i, data_test in enumerate(dataloader_test):

			# Get the batch
			batch, batch_gt = data_test
			# Transform the batch to a float Tensor for the model
			batch = th.Tensor(batch).float().to(device) / 255.0
			batch_gt = th.Tensor(batch_gt).float().to(device)  / 255.0
			batch_gt = batch_gt.unsqueeze(1)

			# Forward pass
			output = model.forward_with_prior(batch)
			# Compute the loss
			error_mask_tiled = th.tile(error_mask, (len(batch_gt),1,1)).unsqueeze(1)
			test_loss += F.mse_loss(output[error_mask_tiled == 1], batch_gt[error_mask_tiled == 1]).item()

		test_loss = test_loss / len(dataloader_test)
			# Add the loss to the running loss
		

	# Save the model if the loss is lower than the previous one
	if epoch == 0:
		min_loss = test_loss
	elif test_loss < min_loss:
		while True:
			try:
				th.save(model.state_dict(), dir_path + '/VAEUnet_{}_test.pth'.format(benchmark))
				print("Best Model saved at epoch {}".format(epoch))
				min_loss = test_loss
				break
			except:
				print("Error while saving the model")
				time.sleep(1)
				continue
	else:
		while True:
			try:
				th.save(model.state_dict(), dir_path + '/VAEUnet_{}_train.pth'.format(benchmark))
				break
			except:
				print("Error while saving the model")
				time.sleep(1)
				continue


	# Add the test loss to the tb writer
	writer.add_scalar('Test/Loss', test_loss, epoch)
	writer.add_scalar('Test/Loss_mse', running_loss_mse[-1], epoch)
	writer.add_scalar('Test/Loss_KL', running_loss_kl[-1], epoch)
	writer.add_scalar('Test/Loss_perceptual', running_loss_perceptual[-1], epoch)

	# Add the loss to the tb writer
	writer.add_scalar('Train/Loss', running_loss[-1], epoch)

	# Print the loss
	print("\nEpoch: {}/{} Train Loss: {:.3f}, Recon loss: {:.3f}, KL loss: {:.3f}, Perceptual loss: {:.3f},  Test Loss: {:.3f}\n".format(epoch, N_epochs, running_loss[-1], running_loss_mse[-1], running_loss_kl[-1], running_loss_perceptual[-1], test_loss))





