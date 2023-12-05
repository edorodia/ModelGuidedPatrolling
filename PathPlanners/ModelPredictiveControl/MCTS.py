"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math


class MCTS:
	"""Monte Carlo tree searcher. First rollout the tree then choose a move."""
	
	def __init__(self, exploration_weight=1, max_depth=10):
		self.Q = defaultdict(int)  # total reward of each node
		self.N = defaultdict(int)  # total visit count for each node
		self.children = dict()  # children of each node
		self.exploration_weight = exploration_weight
		self.max_depth = max_depth
	
	def score(self, node):
		"""Calculate the score of the node"""
		if self.N[node] == 0:
			return float("-inf")
		return self.Q[node] / self.N[node]
	
	def choose(self, node):
		"""Choose the best successor of node. (Choose a move in the game)"""
		if node.is_terminal():
			raise RuntimeError(f"choose called on terminal node {node}")
		
		if node not in self.children:
			return node.find_child_heuristically()
		
		scores = [self.score(n) for n in self.children[node]]
		
		return self.children[node][scores.index(max(scores))]
	
	def choose_terminal(self, node):
		""" Recursively find each best children from node until a terminal state is reached. Return this final node """
		
		path = []
		while True:
			
			path.append(node)
			
			if node not in self.children or not self.children[node]:
				# node is either unexplored or terminal
				return path[-1]
			
			unexplored = self.children[node] - self.children.keys()
			
			if unexplored:
				n = unexplored.pop()
				path.append(n)
				return path[-1]
			
			node = self._greedy_select(node)  # descend a layer deeper
			
	def _greedy_select(self, node):
		""" Select the best child of node. (Choose a move in the game)"""
		
		return max(self.children[node], key=self.score)
		
	def do_rollout(self, node):
		"""Make the tree one layer better. (Train for one iteration.)"""
		path = self._select(node)
		leaf = path[-1]
		self._expand(leaf)
		reward = self._simulate(leaf)
		self._backpropagate(path, reward)
	
	def _select(self, node):
		"""Find an unexplored descendent of `node`"""
		path = []
		while True:
			path.append(node)
			if node not in self.children or not self.children[node]:
				# node is either unexplored or terminal
				return path
			unexplored = self.children[node] - self.children.keys()
			if unexplored:
				n = unexplored.pop()
				path.append(n)
				return path
			node = self._uct_select(node)  # descend a layer deeper
	
	def _expand(self, node):
		"""Update the 'children' dict with the children of 'node' """
		
		if node in self.children:
			return  # already expanded
		
		self.children[node] = node.find_children(node.previous_action)
	
	def _simulate(self, node):
		""" Returns the reward for a random simulation (to completion) of `node` """
		
		total_reward = node.reward
		
		for _ in range(self.max_depth):
			
			if node.is_terminal() or node.depth >= self.max_depth:
				return total_reward
			
			node = node.find_child_heuristically()  # choose a child node using a heuristic
			
			total_reward += node.get_reward()
		
		return total_reward
	
	def _backpropagate(self, path, reward):
		""" Send the reward back up to the ancestors of the leaf """
		for node in reversed(path):
			self.N[node] += 1
			self.Q[node] += reward
	
	def _uct_select(self, node):
		"""Select a child of node, balancing exploration & exploitation"""
		
		# All children of node should already be expanded:
		assert all(n in self.children for n in self.children[node])
		
		log_n_vertex = 2*math.log(self.N[node])
		
		uct_values = [self.uct(n, log_n_vertex) for n in self.children[node]]
		
		return self.children[node][uct_values.index(max(uct_values))]
	
	def uct(self, node, log_n_vertex):
		""" Upper confidence bound for trees """
		
		# Get the reward of the node
		reward = self.Q[node]
		
		# Get the number of visits of the node
		visits = self.N[node]
		
		# Compute the UCB
		if visits == 0:
			ucb_value = float('inf')
		else:
			ucb_value = reward/visits + self.exploration_weight * math.sqrt(log_n_vertex / visits)
		
		return ucb_value
	
	def print_tree(self, node, file=None, _prefix="", _last=True):
		""" Print the tree with the Q/N values of each node"""
		
		if self.N[node] == 0:
			average_value = 0
			print(_prefix, "`- " if _last else "|- ", node.name, sep="", file=file)

		else:
			average_value = self.Q[node] / self.N[node]
			print(_prefix, "`- " if _last else "|- ", node.name, " - Avg.: {0:.5f} ".format(average_value), sep="", file=file)
		_prefix += "   " if _last else "|  "
		
		""" Check if the node has children"""
		if node not in self.children:
			#print(_prefix, "`- ", "Terminal", sep="", file=file)
			return
		else:
			children = self.children[node]
			count = 0
			for i, child in enumerate(children):
				_last = i == (len(children) - 1)
				count += 1
				self.print_tree(child, file, _prefix, _last)
			
			#if count == 0:
				#print(_prefix, "`- ", "Terminal", sep="", file=file)


class Node(ABC):
	"""
	A representation of the game state
	"""
	
	@abstractmethod
	def find_children(self):
		"""All possible successors of this board state"""
		return set()
	
	@abstractmethod
	def find_child_heuristically(self):
		"Heuristic best successor of this board state (for more efficient simulation)"
		
		return None
	
	@abstractmethod
	def find_child_randomly(self):
		"Heuristic best successor of this board state (for more efficient simulation)"
		
		return None
	
	@abstractmethod
	def is_terminal(self):
		"Returns True if the node has no children"
		return True
	
	@abstractmethod
	def get_reward(self):
		"Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
		return 0
	
	@abstractmethod
	def __hash__(self):
		"Nodes must be hashable"
		return 123456789
	
	@abstractmethod
	def __eq__(node1, node2):
		"Nodes must be comparable"
		return True
