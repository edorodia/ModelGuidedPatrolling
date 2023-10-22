"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import sys
sys.path.append('.')

import optuna
from optuna.trial import TrialState
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from argparse import ArgumentParser
from tqdm import tqdm
from ModelTrain.dataset import StaticDataset
from Models.unet import VAEUnet
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import time

import joblib


arg_parser = ArgumentParser()

arg_parser.add_argument('--n_trials', type=int, default=50)
arg_parser.add_argument('--timeout', type=int, default=2000)
arg_parser.add_argument('--cuda', type=int, default=0)
arg_parser.add_argument('--gt', type=str, default='algae_bloom', choices=['algae_bloom', 'shekel'])


args = arg_parser.parse_args()

train_traj_file_name = 'ModelTrain/Data/trajectories_' + args.gt + '_train.npy'
train_gt_file_name = 'ModelTrain/Data/gts_' + args.gt + '_train.npy'
test_traj_file_name = 'ModelTrain/Data/trajectories_' + args.gt + '_test.npy'
test_gt_file_name = 'ModelTrain/Data/gts_' + args.gt + '_test.npy'

# Create the dataset
dataset = StaticDataset(path_trajectories = train_traj_file_name, 
						path_gts = train_gt_file_name,
						transform=None)

dataset_test = StaticDataset(path_trajectories = test_traj_file_name, 
						path_gts = test_gt_file_name,
						transform=None)


# Create the dataloader
BATCHSIZE = 64
dataloader = DataLoader(dataset, batch_size = BATCHSIZE, shuffle = True, num_workers = 0, )
dataloader_test = DataLoader(dataset_test, batch_size = BATCHSIZE, shuffle = True, num_workers = 0)
# Obtain the mask
error_mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
error_mask = torch.Tensor(error_mask).to(device)

dir_path = 'runs/optuna/{}/{}/'.format(args.gt, time.strftime("%Y%m%d-%H%M%S"))

# Create the directory if it does not exist
if not os.path.exists(dir_path):
	os.makedirs(dir_path)
else:
	raise Exception("Directory already exists")


N_epochs = 30


def beta_scheduler(t, beta_min = 0.001, beta_max=1.0, t_beta_min=0.0, t_beta_max=0.2):

	if t < t_beta_min:
		return beta_min
	elif t_beta_min > t > t_beta_max:
		return beta_min + (beta_max - beta_min) * (t - t_beta_min) / (t_beta_max - t_beta_min)
	else:
		return beta_max

def objective(trial):

	# Generate the model.
	input_shape = dataset.trajectories.shape[2:]
	model = VAEUnet(input_shape=input_shape, n_channels_in=2, n_channels_out=1, bilinear=False, scale=2).to(device)

	# Generate the parameters
	
	lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
	L_KL_max = trial.suggest_float("L_KL_max", 0.1, 10.0, log=False)
	L_reconstruction = trial.suggest_float("L_reconstruction", 0.1, 10.0, log=False)
	L_perceptual = trial.suggest_float("L_perceptual", 0.1, 10.0, log=False)

	optimizer = torch.optim.Adam(model.parameters(), lr = lr)

	for epoch in tqdm(range(N_epochs)):

		model.train()

		for i, data in enumerate(dataloader):

			# Get the batch
			batch, batch_gt = data

			# Transform the batch to a float Tensor for the model
			with torch.no_grad():
				batch = torch.Tensor(batch).float().to(device) / 255.0
				batch_gt = torch.Tensor(batch_gt).float().to(device) / 255.0
				batch_gt = batch_gt.unsqueeze(1)

			# Forward pass
			output, prior, posterior = model(batch, batch_gt)

			# Compute the loss
			loss, recon_loss, kl_loss, perceptual_loss = model.compute_loss(x = batch[:,1,:,:], 
														x_true = batch_gt, 
														x_out = output, 
														prior = prior, 
														posterior = posterior, 
														alpha = L_reconstruction,
														beta = beta_scheduler(epoch/N_epochs, beta_max=L_KL_max), 
														gamma = L_perceptual, 
														error_mask=error_mask)

		
			# Reset the gradients
			optimizer.zero_grad()
			# Backward pass
			loss.backward()
			# Apply the gradients
			optimizer.step()

		# Test the model
		model.eval()

		with torch.no_grad():

			# Get the batch

			test_loss = 0  # Test loss to 0

			for i, data_test in enumerate(dataloader_test):

				# Get the batch
				batch, batch_gt = data_test
				# Transform the batch to a float Tensor for the model
				batch = torch.Tensor(batch).float().to(device) / 255.0
				batch_gt = torch.Tensor(batch_gt).float().to(device) / 255.0
				batch_gt = batch_gt.unsqueeze(1)

				# Forward pass
				output = model.forward_with_prior(batch)
				# Compute the loss
				error_mask_tiled = torch.tile(error_mask, (len(batch_gt),1,1)).unsqueeze(1)
				test_loss += F.mse_loss(output[torch.where(error_mask_tiled == 1)], batch_gt[torch.where(error_mask_tiled == 1)]).item()

				# Add the loss to the running loss

			test_loss /= len(dataset_test)

		# Save the model if the loss is lower than the previous one
		if epoch == 0:
			min_loss = test_loss
		elif test_loss < min_loss:
			while True:
				try:
					torch.save(model.state_dict(), dir_path + 'VAEUnet_{}_test_trial_num_{}.pth'.format(args.gt, trial.number))
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
					torch.save(model.state_dict(), dir_path + 'VAEUnet_{}_train_trial_num_{}.pth'.format(args.gt,trial.number))
					break
				except:
					print("Error while saving the model")
					time.sleep(1)
					continue




		# Print the loss
		trial.report(min_loss, epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	return test_loss


if __name__ == "__main__":
	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=30)

	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))

	## Save the study
	study_name = dir_path + 'optuna_{}_{}'.format(args.gt, time.strftime("%Y%m%d-%H%M%S"))

	# Save using joblib
	joblib.dump(study, study_name)


	