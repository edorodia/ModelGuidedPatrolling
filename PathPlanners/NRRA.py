import numpy as np
import pandas as pd 
from tqdm import trange
import matplotlib.pyplot as plt


class WanderingAgent:

	def __init__(self, world: np.ndarray, movement_length: float, number_of_actions: int, consecutive_movements = None, seed = 0):
		
		self.world = world
		self.move_length = movement_length
		self.number_of_actions = number_of_actions
		self.consecutive_movements = consecutive_movements
		self.t = 0
		self.action = None
		self.seed = seed
	
	def move(self, actual_position):

		#if no action was already performed selects one randomly
		if self.action is None:
			self.action = self.select_action_without_collision(actual_position)
		
		# Compute if there is an obstacle or reached the border #
		# checks if the action made previously will bring to a collision otherwise it keeps that action valid
		OBS = self.check_collision(self.action, actual_position)

		#if the action made previously isn't anymore valid, it chooses another one, excluding collision generating ones and the opposite
		if OBS:
			self.action = self.select_action_without_collision(actual_position)

		#consecutive_movements indicates how many steps can be performed without changing the trajectory of the agent
		if self.consecutive_movements is not None:
			if self.t == self.consecutive_movements:
				self.action = self.select_action_without_collision(actual_position)
				self.t = 0

		self.t += 1
		return self.action
	
	
	def action_to_vector(self, action):
		""" Transform an action to a vector """

		#2*pi*(1/8) -> divides the 360 degrees angle in eight equal portions calculating the cos and sin of every angle and putting it in the vectors array
		#cos -> what has to be added to the X position in order to move horizontally
		#sin -> what has to be added to the Y position in order to move vertically
		#this data is obviously calculated on a radius_of_circle = 1 
		vectors = np.array([[np.cos(2*np.pi*i/self.number_of_actions), np.sin(2*np.pi*i/self.number_of_actions)] for i in range(self.number_of_actions)])

		#this means that the action given in input is an integer number with values allowed from 0 to (self.number_of_actions - 1)
		return np.round(vectors[action]).astype(int)
	
	def opposite_action(self, action):
		""" Compute the opposite action """
		return (action + self.number_of_actions//2) % self.number_of_actions
	
	def check_collision(self, action, actual_position):
		""" Check if the agent collides with an obstacle """
		new_position = actual_position + self.action_to_vector(action) * self.move_length
		new_position = np.ceil(new_position).astype(int)
		
		OBS = (new_position[0] < 0) or (new_position[0] >= self.world.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.world.shape[1])
		if not OBS:
			OBS = self.world[new_position[0], new_position[1]] == 0

		return OBS

	def select_action_without_collision(self, actual_position):
		""" Select an action without collision """
		#has a true value for every action that brings to a collision
		action_caused_collision = [self.check_collision(action, actual_position) for action in range(self.number_of_actions)]

		# Select a random action without collision and that is not the oppositve previous action #
		if self.action is not None:
			#checks if a previous action exists, if it does just sets the collision of that action to True so that it can be avoided as the others
			opposite_action = self.opposite_action(self.action)
			action_caused_collision[opposite_action] = True
		#selects an action from the inverted action_caused_collision, now the true values are where the action can be performed 
		action = np.random.choice(np.where(np.logical_not(action_caused_collision))[0])

		return action
	
