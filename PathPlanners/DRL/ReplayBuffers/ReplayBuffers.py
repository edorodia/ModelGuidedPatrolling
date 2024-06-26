import numpy as np
from collections import deque
from typing import Deque, Dict, Tuple, List, Union, Any
from ..ReplayBuffers.utils import MinSegmentTree, SumSegmentTree
import random

class ReplayBuffer:
	"""A simple numpy replay buffer."""

	#n_step: indicates the number of experiences to aggregate
	def __init__(self, obs_dim: Union[tuple, int, list], size: int, batch_size: int = 32, n_step: int = 1, gamma: float = 0.99, obs_dtype=np.float32):

		self.obs_buf = np.zeros([size] + list(obs_dim), dtype=obs_dtype)
		self.next_obs_buf = np.zeros([size] + list(obs_dim), dtype=obs_dtype)
		self.acts_buf = np.zeros([size], dtype=np.float32)
		self.rews_buf = np.zeros([size], dtype=np.float32)
		self.done_buf = np.zeros(size, dtype=np.float32)
		self.info_buf = np.empty([size], dtype=dict)
		self.max_size, self.batch_size = size, batch_size
		self.ptr, self.size, = 0, 0

		# for N-step Learning
		#deque is a double-ended queue, similar to a list but way more optimized
		self.n_step_buffer = deque(maxlen=n_step)
		self.n_step = n_step
		self.gamma = gamma

	def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool, info: dict) -> Tuple[
		np.ndarray, np.ndarray, float, np.ndarray, bool, dict]:

		transition = (obs, act, rew, next_obs, done, info)
		self.n_step_buffer.append(transition)

		# single step transition is not ready
		if len(self.n_step_buffer) < self.n_step:
			return ()

		# make a n-step transition, aggregates values of the last n_step experiences
		rew, next_obs, done, info = self._get_n_step_info(self.n_step_buffer, self.gamma)
		#obs and act related to the n_step experiences are all linked to the first experience
		obs, act = self.n_step_buffer[0][:2]

		self.obs_buf[self.ptr] = obs
		self.next_obs_buf[self.ptr] = next_obs
		self.acts_buf[self.ptr] = act
		self.rews_buf[self.ptr] = rew
		self.done_buf[self.ptr] = done
		self.info_buf[self.ptr] = info
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

		return self.n_step_buffer[0]

	def sample_batch(self) -> Dict[str, np.ndarray]:
		idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

		return dict(
			obs=self.obs_buf[idxs],
			next_obs=self.next_obs_buf[idxs],
			acts=self.acts_buf[idxs],
			rews=self.rews_buf[idxs],
			done=self.done_buf[idxs],
			info=self.info_buf[idxs],
			# for N-step Learning
			indices=idxs,
		)

	def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
		# for N-step Learning

		return dict(
			obs=self.obs_buf[idxs],
			next_obs=self.next_obs_buf[idxs],
			acts=self.acts_buf[idxs],
			rews=self.rews_buf[idxs],
			done=self.done_buf[idxs],
			info=self.info_buf[idxs],
		)

	@staticmethod
	def _get_n_step_info(n_step_buffer: Deque, gamma: float) -> Tuple[np.int64, np.ndarray, bool, dict]:
		"""Return n step rew, next_obs, and done."""
		# info of the last transition
		#extracts reward next_state, done flag of the last experience
		rew, next_obs, done, info = n_step_buffer[-1][-4:]

		#iters over all the transitions from the penultimate back to the first and upgrades the reward found with all the experience observed
		for transition in reversed(list(n_step_buffer)[:-1]):
			r, n_o, d = transition[-3:]
			rew = r + gamma * rew * (1 - d)
			next_obs, done = (n_o, d) if d else (next_obs, done)

		return rew, next_obs, done, info

	def __len__(self) -> int:
		return self.size

#tipically priority has to be assigned to experiences with an higher temporal difference, cause these are the ones that increase the efficiency of parameteres optimization
class PrioritizedReplayBuffer(ReplayBuffer):
	"""Prioritized Replay buffer.

	Attributes:
		max_priority (float): max priority
		tree_ptr (int): next index of tree
		alpha (float): alpha parameter for prioritized replay buffer
		sum_tree (SumSegmentTree): sum tree for prior
		min_tree (MinSegmentTree): min tree for min prior to get max weight

	"""

	def __init__(self, obs_dim: Union[tuple, int, list], size: int, batch_size: int = 32, alpha: float = 0.6, n_step: int = 1, gamma: float = 0.99, obs_dtype=np.float32):
		"""Initialization."""
		assert alpha >= 0

		super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size, n_step, gamma, obs_dtype)

		self.max_priority, self.tree_ptr = 1.0, 0
		self.alpha = alpha

		# capacity must be positive and a power of 2.
		tree_capacity = 1
		while tree_capacity < self.max_size:
			tree_capacity *= 2

		self.sum_tree = SumSegmentTree(tree_capacity)
		self.min_tree = MinSegmentTree(tree_capacity)

	def store(
			self,
			obs: np.ndarray,
			act: int,
			rew: float,
			next_obs: np.ndarray,
			done: bool,
			info: dict,
	) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, dict]:
		"""Store experience and priority."""
		transition = super().store(obs, act, rew, next_obs, done, info)

		if transition:
			self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
			self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
			self.tree_ptr = (self.tree_ptr + 1) % self.max_size

		return transition

	def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
		"""Sample a batch of experiences."""
		assert len(self) >= self.batch_size
		assert beta > 0

		indices = self._sample_proportional()

		obs = self.obs_buf[indices]
		next_obs = self.next_obs_buf[indices]
		acts = self.acts_buf[indices]
		rews = self.rews_buf[indices]
		done = self.done_buf[indices]
		info = self.info_buf[indices]
		weights = np.array([self._calculate_weight(i, beta) for i in indices])

		return dict(
			obs=obs,
			next_obs=next_obs,
			acts=acts,
			rews=rews,
			done=done,
			info=info,
			weights=weights,
			indices=indices,
		)

	def update_priorities(self, indices: List[int], priorities: np.ndarray):
		"""Update priorities of sampled transitions."""
		assert len(indices) == len(priorities)

		for idx, priority in zip(indices, priorities):
			assert priority > 0
			assert 0 <= idx < len(self)

			self.sum_tree[idx] = priority ** self.alpha
			self.min_tree[idx] = priority ** self.alpha

			self.max_priority = max(self.max_priority, priority)

	def _sample_proportional(self) -> List[int]:
		"""Sample indices based on proportions."""
		indices = []
		#total weight of experiences in the buffer
		p_total = self.sum_tree.sum(0, len(self) - 1)
		#calculates the mean that should have the group of experiences taken
		segment = p_total / self.batch_size

		for i in range(self.batch_size):
			a = segment * i
			b = segment * (i + 1)
			upperbound = random.uniform(a, b)
			idx = self.sum_tree.retrieve(upperbound)
			indices.append(idx)

		return indices
	
	#beta is an hyperparameter used to regulate the emphasis on high-priority experiences, the higher is beta the higher is the emphasis on high-priority experiences
	def _calculate_weight(self, idx: int, beta: float):
		"""Calculate the weight of the experience at idx."""
		# get max weight
  		#this op is used to normalize the result
		p_min = self.min_tree.min() / self.sum_tree.sum()
		max_weight = (p_min * len(self)) ** (-beta)

		# calculate weights
		p_sample = self.sum_tree[idx] / self.sum_tree.sum()
		weight = (p_sample * len(self)) ** (-beta)
		weight = weight / max_weight

		return weight





