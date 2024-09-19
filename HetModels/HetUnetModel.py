import sys

import torch
import torch as th
import numpy as np
from Models.unet import UNet, VAEUnet
from HetModels.HetMiopicModel import HetMiopicModel


benchmark_2_vae_path_het = {'algae_bloom': r'runs/optuna/algae_bloom/VAEUnet_algae_bloom_test_trial_num_12.pth',
                        'shekel':      r'runs/optuna/shekel/VAEUnet_shekel_test_trial_num_18.pth'}


class UnetDeepHetModel:

	def __init__(self, navigation_map: np.ndarray,
	             model_path: str,
	             device: str = 'cuda:0',
	             resolution=1,
	             influence_radius=2, dt=0.7):
		
		self.navigation_map = navigation_map
		self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

		# Create the miopic predictor
		self.pre_model = HetMiopicModel(navigation_map, influence_radius, resolution, dt)
		# Create the model
		self.model = UNet(n_channels_in=2, n_channels_out=1, bilinear=False, scale=2).to(device)
		# Import the model (loads a pretrained model state dictionary from file)
		self.model.load_state_dict(th.load(model_path))
		#Sets the model in evaluation mode
		self.model.eval()

		# Sets the map where to work, the navigation map
		self.model_map = np.zeros_like(self.navigation_map)

	def update(self, from_ASV: bool = False, from_Drone: bool = False, ASV_positions: np.ndarray = None, ASV_values: np.ndarray = None, Drone_positions: np.ndarray = None, Drone_values: np.ndarray = None, ASV_visited_map: np.ndarray = None, Drone_visited_map: np.ndarray = None):
		
		# Update the het miopic model
		self.pre_model.update(from_ASV = from_ASV, from_Drone = from_Drone, ASV_positions = ASV_positions, ASV_values = ASV_values, Drone_positions = Drone_positions, Drone_values = Drone_values)

		# Use the miopic model to predict the map
		pre_model_map = self.pre_model.predict()

		with th.no_grad():
			# Feed the model
			# Visited_map
			input_tensor_0 = th.from_numpy(np.logical_or(ASV_visited_map, Drone_visited_map)).unsqueeze(0).unsqueeze(0).to(self.device).float()
			# Knowledge acquired till now by the ASVs
			input_tensor_1 = th.from_numpy(pre_model_map).unsqueeze(0).unsqueeze(0).to(self.device).float()
			# Stack using the dim 1
			input_tensor = th.cat((input_tensor_0, input_tensor_1), dim=1)
			# Predict the model #
			output_tensor = self.model(input_tensor)
			# Get the numpy array
			model_map = output_tensor.squeeze(0).squeeze(0).cpu().detach().numpy() * self.navigation_map
			# overrides with the correct data in the already explored parts
			model_map[self.pre_model.x[:, 0], self.pre_model.x[:, 1]] = self.pre_model.y
			self.model_map = model_map

	def predict(self):
		return self.model_map

	def reset(self):
		self.model_map = np.zeros_like(self.navigation_map)
		self.pre_model.reset()


class VAEUnetDeepHetModel(UnetDeepHetModel):
	""" Subclass of UnetDeepModel that uses a VAEUnet model """

	def __init__(self, navigation_map: np.ndarray, model_path: str, device: str = 'cuda:0', resolution=1,
	             influence_radius=2, dt=0.7, N_imagined=1):

		self.navigation_map = navigation_map
		self.device = th.device(device if th.cuda.is_available() else 'cpu')

		# Create the miopic predictor
		self.pre_model = HetMiopicModel(navigation_map, influence_radius, resolution, dt)
		# Create the models
		self.model = VAEUnet(input_shape=navigation_map.shape, n_channels_in=2, n_channels_out=1, bilinear=False, scale=2).to(device)
	
		
		self.model.eval()
		# Import the model
		self.model.load_state_dict(th.load(model_path, map_location=device))
		self.model.eval()

		self.model_map = np.zeros_like(self.navigation_map)

		self.N_imagined = N_imagined

	def update(self, from_ASV: bool = False, from_Drone: bool = False, ASV_positions: np.ndarray = None, ASV_values: np.ndarray = None, Drone_positions: np.ndarray = None, Drone_values: np.ndarray = None, ASV_visited_map: np.ndarray = None, Drone_visited_map: np.ndarray = None):

		# Update the het miopic model
		self.pre_model.update(from_ASV = from_ASV, from_Drone = from_Drone, ASV_positions = ASV_positions, ASV_values = ASV_values, Drone_positions = Drone_positions, Drone_values = Drone_values)

		# Use the miopic model to predict the map
		pre_model_map = self.pre_model.predict()

		with th.no_grad():

			# Feed the model
			input_tensor_0 = th.from_numpy(np.logical_or(ASV_visited_map, Drone_visited_map)).unsqueeze(0).unsqueeze(0).to(self.device).float()
			input_tensor_1 = th.from_numpy(pre_model_map).unsqueeze(0).unsqueeze(0).to(self.device).float()
			# Stack using the dim 1
			input_tensor = th.cat((input_tensor_0, input_tensor_1), dim=1)
			# Predict the model #
			if self.N_imagined == 1:
				output_tensor = self.model.forward_with_prior(input_tensor)
			else:
				output_tensor = self.model.imagine(N=self.N_imagined, x=input_tensor)

			# Get the numpy array

			self.model_map = output_tensor.squeeze(0).squeeze(0).cpu().detach().numpy() * self.navigation_map
