from HetModels.NoiseModels.BaseNoiseModel import BaseNoiseModel
from HetModels.NoiseModels.MeanNoise import MeanNoise
import numpy as np

class FishEyeNoiseApproximator(BaseNoiseModel):

    def __init__(self, influence_side: float, internal_true_to_cell_side: float):
        self.influence_side = int(influence_side)

        self.switch_model = False

        # In this case the internal square can exist but check for the side to be odd (otherwise it won't be centerable) #
        if internal_true_to_cell_side <= influence_side-2:

            if internal_true_to_cell_side % 2 == 0:
                self.internal_true_to_cell_side = internal_true_to_cell_side + 1
            else:
                self.internal_true_to_cell_side = internal_true_to_cell_side

        # If the internal side is higher than influence_side - 1 then just call the normal MeanNoise #
        elif internal_true_to_cell_side >= influence_side-1:

            self.internal_true_to_cell_side = influence_side
            self.switch_model = True
        
        self.internal_true_to_cell_side = int(self.internal_true_to_cell_side)

    # Returns the matrix without editing anything at all #
    def mask(self, positions: np.ndarray, values: np.ndarray):
        if self.switch_model:
            # Use the MeanNoise model #
            return MeanNoise().mask(positions, values)
        else:
            # Use the new filter #
            square = self._generate_square_from_positions(positions, values)

            original_starting_position = np.array([np.min(positions[:,0]), np.min(positions[:,1])])

            side_mean_squares = int(np.floor((self.influence_side - self.internal_true_to_cell_side) / 2))

            slide_begin = 0 + side_mean_squares + self.internal_true_to_cell_side

            # Calculate the mean angle cells #
            top_left_square = square[0:0 + side_mean_squares, 0:0 + side_mean_squares]
            tl_positions, tl_values = self._generate_pos_values_from_square(top_left_square)
            tl_positions, tl_values = MeanNoise().mask(tl_positions, tl_values)
            tl_positions = self.shift_columns(original_starting_position[1], tl_positions)
            tl_positions = self.shift_rows(original_starting_position[0], tl_positions)

            top_right_square = square[0:0 + side_mean_squares, slide_begin: self.influence_side]
            tr_positions, tr_values = self._generate_pos_values_from_square(top_right_square)
            tr_positions, tr_values = MeanNoise().mask(tr_positions, tr_values)
            tr_positions = self.shift_columns(original_starting_position[1]+side_mean_squares+self.internal_true_to_cell_side, tr_positions)
            tr_positions = self.shift_rows(original_starting_position[0], tr_positions)


            bottom_left_square = square[slide_begin: self.influence_side, 0:0 + side_mean_squares]
            bl_positions, bl_values = self._generate_pos_values_from_square(bottom_left_square)
            bl_positions, bl_values = MeanNoise().mask(bl_positions, bl_values)
            bl_positions = self.shift_columns(original_starting_position[1], bl_positions)
            bl_positions = self.shift_rows(original_starting_position[0]+side_mean_squares+self.internal_true_to_cell_side, bl_positions)

            bottom_right_square = square[slide_begin: self.influence_side, slide_begin: self.influence_side]
            br_positions, br_values = self._generate_pos_values_from_square(bottom_right_square)
            br_positions, br_values = MeanNoise().mask(br_positions, br_values)
            br_positions = self.shift_columns(original_starting_position[1]+side_mean_squares+self.internal_true_to_cell_side, br_positions)
            br_positions = self.shift_rows(original_starting_position[0]+side_mean_squares+self.internal_true_to_cell_side, br_positions)

            # Calculate the mean side cells #
            top_side = square[0: side_mean_squares, side_mean_squares: side_mean_squares + self.internal_true_to_cell_side ]
            t_positions, t_values = self._generate_pos_values_from_square(top_side)
            t_positions, t_values = MeanNoise().mask(t_positions, t_values)
            t_positions = self.shift_columns(original_starting_position[1]+side_mean_squares, t_positions)
            t_positions = self.shift_rows(original_starting_position[0], t_positions)

            left_side = square[side_mean_squares: side_mean_squares + self.internal_true_to_cell_side, 0: side_mean_squares ]
            l_positions, l_values = self._generate_pos_values_from_square(left_side)
            l_positions, l_values = MeanNoise().mask(l_positions, l_values)
            l_positions = self.shift_columns(original_starting_position[1], l_positions)
            l_positions = self.shift_rows(original_starting_position[0]+side_mean_squares, l_positions)

            bottom_side = square[side_mean_squares + self.internal_true_to_cell_side: self.influence_side, side_mean_squares: side_mean_squares + self.internal_true_to_cell_side]
            b_positions, b_values = self._generate_pos_values_from_square(bottom_side)
            b_positions, b_values = MeanNoise().mask(b_positions, b_values)
            b_positions = self.shift_columns(original_starting_position[1]+side_mean_squares, b_positions)
            b_positions = self.shift_rows(original_starting_position[0]+side_mean_squares+self.internal_true_to_cell_side, b_positions)

            right_side = square[side_mean_squares: side_mean_squares + self.internal_true_to_cell_side, side_mean_squares + self.internal_true_to_cell_side: self.influence_side]
            r_positions, r_values = self._generate_pos_values_from_square(right_side)
            r_positions, r_values = MeanNoise().mask(r_positions, r_values)
            r_positions = self.shift_columns(original_starting_position[1]+side_mean_squares+self.internal_true_to_cell_side, r_positions)
            r_positions = self.shift_rows(original_starting_position[0]+side_mean_squares, r_positions)

            # the center cell remains the same #
            center_cells = square[side_mean_squares: side_mean_squares + self.internal_true_to_cell_side, side_mean_squares: side_mean_squares + self.internal_true_to_cell_side]
            center_positions, center_values = self._generate_pos_values_from_square(center_cells)
            center_positions = self.shift_columns(original_starting_position[1]+side_mean_squares, center_positions)
            center_positions = self.shift_rows(original_starting_position[0]+side_mean_squares, center_positions)

            final_positions = np.concatenate((tl_positions, tr_positions, bl_positions, br_positions, t_positions, l_positions, b_positions, r_positions, center_positions))
            final_values = np.concatenate((tl_values,    tr_values,    bl_values,    br_values,    t_values,    l_values,    b_values,    r_values,    center_values))

            #gets the indices of the final_positions sorted by rows first and then columns
            """ firstly we would have 3 ... , 3 ..., 3..., 4 ..., 4 ..., and so on with all the rows sorted but columns still shuffled, so then we sort by columns to  """
            sorted_indices = np.lexsort((final_positions[:,1], final_positions[:,0]))
            # Use the sorted indices to sort the positions and values arrays
            sorted_positions = final_positions[sorted_indices]
            sorted_values = final_values[sorted_indices]

            return sorted_positions, sorted_values


