import numpy as np

class BaseNoiseModel:

	def __init__(self) -> None:
		pass

	"""
	applies the mask to the positions of the square given in input
	"""
	def mask(positions: np.ndarray, values: np.ndarray):
		pass

	def _generate_square(positions: np.ndarray, values: np.ndarray):
		# Get the smallest and biggest value for every column (for rows and columns of the square)
		min_row = np.min(positions, axis=0)[0]
		min_column = np.min(positions, axis=0)[1]
		max_row = np.max(positions, axis=0)[0]
		max_column = np.max(positions, axis=0)[1]

		square = np.zeros(((max_row-min_row) + 1, (max_column-min_column) + 1), dtype=float)

		square[positions[:,0], positions[:,1]] = values

		return square