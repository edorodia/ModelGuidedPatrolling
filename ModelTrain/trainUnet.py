import sys 
sys.path.append('.')
from ModelTrain.dataset import StaticDataset
from Models.unet import UNet
import torch as th
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
argparser.add_argument('--epochs', type=int, default=10)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--lr', type=float, default=1e-4)
argparser.add_argument('--weight_decay', type=float, default=1e-5)
argparser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cuda:1', 'cpu']) 

args = argparser.parse_args()

benchmark = args.benchmark
train_traj_file_name = 'ModelTrain/Data/trajectories_' + benchmark + '_train.npy'
train_gt_file_name = 'ModelTrain/Data/gts_' + benchmark + '_train.npy'
test_traj_file_name = 'ModelTrain/Data/trajectories_' + benchmark + '_test.npy'
test_gt_file_name = 'ModelTrain/Data/gts_' + benchmark + '_test.npy'

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
model = UNet(n_channels_in=2, n_channels_out=1).to(device)

# Define the optimizer
optimizer = th.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
#NOTE: The loss function is defined in the model

# Define the number of epochs
N_epochs = args.epochs

# Start the training loop

# Remove the previous tensorboard log and create a new one with the current time
dir_path = 'runs/TrainingUnet/Unet_{}_{}'.format(benchmark, time.strftime("%Y%m%d-%H%M%S"))
os.system('rm -rf ' + dir_path)
writer = tb.SummaryWriter(log_dir=dir_path, comment='Unet_training_{}'.format(benchmark))

mask_tensor = th.Tensor(mask).float().to(device)

pbar_epoch = tqdm(total=N_epochs, desc="Epoch progress: ")
pbar_batch = tqdm(total=len(dataloader) // args.batch_size, desc="Batch progress: ")

for epoch in tqdm(range(N_epochs), desc="Epochs: "):

	running_loss = []
	pbar_epoch.update(1)
	model.train()

	for i, data in tqdm(enumerate(dataloader), desc="Batches: ")

		pbar_batch

		# Get the batch
		batch, batch_gt = data

		# Transform the batch to a float Tensor for the model
		batch = th.Tensor(batch).float().to(device)
		batch_gt = th.Tensor(batch_gt).float().to(device)
		batch_gt = batch_gt.unsqueeze(1)


		# Reset the gradients
		optimizer.zero_grad()

		# Forward pass
		output = model(batch)

		# Compute the loss
		loss = model.compute_loss(x_predicted=output, x_gt=batch_gt, mask=mask_tensor)

		# Add the loss to the running loss
		running_loss.append(loss.item())
		
		# Backward pass
		loss.backward()
		# Apply the gradients
		optimizer.step()

	# Test the model
	model.eval()

	with th.no_grad():

		running_test_loss = []

		# Get the batch

		for i, data_test in enumerate(dataloader_test):

			# Get the batch
			batch, batch_gt = data_test
			# Transform the batch to a float Tensor for the model
			batch = th.Tensor(batch).float().to(device)
			batch_gt = th.Tensor(batch_gt).float().to(device)
			batch_gt = batch_gt.unsqueeze(1)

			# Forward pass
			output = model(batch)
			# Compute the loss
			test_loss = model.compute_loss(x_predicted=output, x_gt=batch_gt, mask=mask_tensor)

			# Add the loss to the running loss
			running_test_loss.append(test_loss.item())

	# Save the model if the loss is lower than the previous one
	if epoch == 0:
		min_loss = np.mean(running_test_loss)
	elif np.mean(running_test_loss) < min_loss:
		th.save(model.state_dict(), dir_path + '/Unet_{}_test.pth'.format(benchmark))
		print("Model saved at epoch {}".format(epoch))
		min_loss = np.mean(running_test_loss)
	else:
		th.save(model.state_dict(), dir_path + '/Unet_{}_train.pth'.format(benchmark))


	# Add the test loss to the tb writer
	writer.add_scalar('Test/Loss', np.mean(running_test_loss), epoch)

	# Add the loss to the tb writer
	writer.add_scalar('Train/Loss', np.mean(running_loss), epoch)

	# Add the images to the tb writer

	# Get the first image of the batch
	test_input = batch[0, 1, :, :].unsqueeze(0)
	test_gt = batch_gt[0, 0, :, :].unsqueeze(0)
	test_output = output[0, 0, :, :].unsqueeze(0)

	# Add the images to the tb writer	
	writer.add_image('Test/Input', test_input, epoch)
	writer.add_image('Test/GroundTruth', test_gt, epoch)
	writer.add_image('Test/Prediction', test_output, epoch)

	# Print the loss
	print("Epoch: {}/{} Total Loss: {}".format(epoch, N_epochs, np.mean(running_test_loss)))





