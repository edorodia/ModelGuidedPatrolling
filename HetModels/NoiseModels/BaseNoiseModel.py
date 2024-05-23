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
	def mask(self, positions: np.ndarray, values: np.ndarray):
		pass

	"""
	utility function to put all the values of the positions given in input into a new 2d matrix to facilitate the calculation
	"""
	def _generate_square_from_positions(self, positions: np.ndarray, values: np.ndarray):
		# Get the smallest and biggest value for every column (for rows and columns of the square)
		min_row = np.min(positions, axis=0)[0]
		min_column = np.min(positions, axis=0)[1]
		max_row = np.max(positions, axis=0)[0]
		max_column = np.max(positions, axis=0)[1]

		square = np.zeros(((max_row-min_row) + 1, (max_column-min_column) + 1), dtype=float)

		square[positions[:,0] - min_row, positions[:,1] - min_column] = values

		return square

	"""
	utility function to convert a 2d square to an array of positions and an array of corresponding values
	"""
	def _generate_pos_values_from_square(self, square: np.ndarray):
		positions = []
		values = []
		row_id = 0
		column_id = 0
		for row in square:
			for ele in row:
				positions.append(np.array([row_id, column_id]))
				values.append(square[row_id, column_id])
				column_id += 1
			column_id = 0
			row_id += 1
		return np.array(positions), np.array(values)
			
	"""
	utility function to add a value to all the rows in the positions array
	"""
	def shift_rows(self, shift: int, positions: np.ndarray):
		new_positions = []
		for position in positions:
			new_positions.append(np.array([position[0] + shift, position[1]]))
		return np.array(new_positions)
	

	"""
	utility function to add a value to all the columns in the positions array
	"""
	def shift_columns(self, shift: int, positions: np.ndarray):
		new_positions = []
		for position in positions:
			new_positions.append(np.array([position[0], position[1] + shift]))
		return np.array(new_positions)