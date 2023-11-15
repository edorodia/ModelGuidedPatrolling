import numpy as np
import pickle
from scipy.interpolate import interp1d


class preComputedExplorationPolicy:
	
	def __init__(self, path_to_trajectories: str, n_agents: int):
		with open(path_to_trajectories, 'rb') as handle:
			self.paths = pickle.load(handle)
		
		assert len(self.paths) == n_agents, "These paths are for a different number of agents"
		
		self.n_agents = n_agents
		
		self.initial_positions = np.asarray([self.paths[0][0], self.paths[1][0], self.paths[2][0], self.paths[3][0]])
		
		for i in range(self.n_agents):
			x = np.arange(len(self.paths[i]))
			f = interp1d(x, self.paths[i], axis=0)
			xnew = np.arange(0, len(self.paths[i]) - 1, 0.5)
			self.paths[i] = f(xnew)
		
		# Remove the last element of the paths
		
		# Remove the last element of the paths
		for i in range(self.n_agents):
			self.paths[i] = self.paths[i][:-1]
		
		self.current_step = 0
	
	def suggest_action(self):
		
		actions = {i: self.paths[i][(self.current_step + 1) % len(self.paths[i])] for i in range(self.n_agents)}
		
		self.current_step += 1
		
		return actions
	
	def reset(self):
		
		self.current_step = 0
