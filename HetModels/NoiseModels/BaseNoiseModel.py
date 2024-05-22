import numpy as np

class BaseNoiseModel:

	def __init__(self) -> None:
		pass

	"""
	applies the mask to the positions of the square given in input
	the input is made of 
	- positions -> an array with the [row,column] position
	- values -> an array with the values read from the ground truth in corresponding position

	the output of this function is of the same type and structure of the input
	- positions -> the same array given in input
	- values -> the values after the noise applied to the square of cells read by the drone
	"""
	def mask(positions: np.ndarray, values: np.ndarray):
		pass

	"""
	utility function to put all the values of the positions given in input into a new 2d matrix to facilitate the calculation
	"""
	def _generate_square(positions: np.ndarray, values: np.ndarray):
		# Get the smallest and biggest value for every column (for rows and columns of the square)
		min_row = np.min(positions, axis=0)[0]
		min_column = np.min(positions, axis=0)[1]
		max_row = np.max(positions, axis=0)[0]
		max_column = np.max(positions, axis=0)[1]

		square = np.zeros(((max_row-min_row) + 1, (max_column-min_column) + 1), dtype=float)

		square[positions[:,0] - min_row, positions[:,1] - min_column] = values

		return square