
import numpy as np
from deap import benchmarks

import sys

sys.path.append('.')

class shekel(object):

    """ Ground Truth generator class.
        It creates a ground truth within the specified navigation map.
        The ground truth is generated randomly following some realistic rules of the enviornment
        and using a Shekel function.
    """

    def __init__(self, grid, max_number_of_peaks=None, is_bounded = True, seed = 0, dt = 0.01):

        """ Maximum number of peaks encountered in the scenario. """
        self.max_number_of_peaks = 6 if max_number_of_peaks is None else max_number_of_peaks
        self.seed = seed
        #np.random.seed(self.seed)
        self.fig = None

        """ random map features creation """
        self.grid = 1.0 - grid.astype(float)
        #creates a grid with 10 cells of height and proportional width
        self.xy_size = np.array([self.grid.shape[1]/self.grid.shape[0]*10, 10])
        self.is_bounded = is_bounded
        self.dt = dt

        #This section of code generates positions of the peaks, and their height
        # Peaks positions bounded from 1 to 9 in every axis
        self.number_of_peaks = np.random.randint(1, self.max_number_of_peaks+1)
        #Generates a set of number_of_peaks couples of values all in the [0,1)
        self.A = np.random.rand(self.number_of_peaks, 2) * self.xy_size # * 0.8 + self.xy_size*0.2
        # Peaks size bounded from a minimum 2.5 to 5
        self.C = 2*np.random.rand(self.number_of_peaks, 1) + 0.5

        """
        print("peaks position -> " + str(self.A))
        print("peaks height -> " + str(self.C))
        """

        """ Creation of the map field """
        self._x = np.arange(0, self.grid.shape[1], 1)
        self._y = np.arange(0, self.grid.shape[0], 1)

        #returns two arrays which, if put together, give in output all the possible combinations of the values proposed
        self._x, self._y = np.meshgrid(self._x, self._y)

        self._z, self.meanz, self.stdz, self.normalized_z = None, None, None, None # To instantiate attr after assigning in __init__
        self.create_field()  # This method creates the normalized_z values

    #calculates the shekel function on every point made up by two dimensions given in input
    #the shekel function used knows how many peaks are there cause they are assumed by the size of the self.A and self.C input
    def shekel_arg0(self, sol):

        return np.nan if self.grid[sol[1]][sol[0]] == 1 else \
            benchmarks.shekel(sol[:2]/np.array(self.grid.shape)*10, self.A, self.C)[0]
        #this function takes in input the following things
        #the list of positions of the peaks
        #the list of values of the peaks
        #the values of x and y of the grid if they are not an obstacle proportioned to the grid scale used

    def create_field(self):

        """ Creation of the normalized z field """
        #in the count phase _x is used alone because both the _y and _x in the way they would get used would be the same
        #reshape allows to go back from single flat dimension to grid dimension like the shape of _x
        self._z = np.fromiter(map(self.shekel_arg0, zip(self._x.flat, self._y.flat)), dtype=np.float32,
                              count=self._x.shape[0] * self._x.shape[1]).reshape(self._x.shape)

        #so now self_z is the grid with all the computed shekel values

        self.meanz = np.nanmean(self._z)
        self.stdz = np.nanstd(self._z)

        if self.stdz > 0.001:
            self.normalized_z = (self._z - self.meanz) / self.stdz
        else:
            self.normalized_z = self._z

        if self.is_bounded:
            self.normalized_z = np.nan_to_num(self.normalized_z, nan=np.nanmin(self.normalized_z))
            self.normalized_z = (self.normalized_z - np.min(self.normalized_z))/(np.max(self.normalized_z) - np.min(self.normalized_z))

    def reset(self):
        """ Reset ground Truth """
        # Peaks positions bounded from 1 to 9 in every axis
        self.number_of_peaks = np.random.randint(1,self.max_number_of_peaks+1)
        self.A = np.random.rand(self.number_of_peaks, 2) * self.xy_size # * 0.9 + self.xy_size*0.1
        # Peaks size bounded from a minimum 2.5 to 5
        self.C = 2*np.random.rand(self.number_of_peaks, 1) + 0.5
        
        # Reconstruct the field #
        self.create_field()

    def read(self, position=None):

        """ Read the complete ground truth or a certain position """

        if position is None:
            return self.normalized_z
        else:

            # Extract rows and columns from positions
            rows = position[:, 0]
            cols = position[:, 1]

            # Check if the indices are within bounds
            row_mask = (rows >= 0) & (rows < self.normalized_z.shape[0])
            col_mask = (cols >= 0) & (cols < self.normalized_z.shape[1])

            # Combined mask for both row and column bounds
            valid_mask = row_mask & col_mask

            # Initialize result array with zeros
            result = np.zeros(len(position))

            # Extract valid indices
            valid_rows = rows[valid_mask]
            valid_cols = cols[valid_mask]

            # Use valid indices to fetch values from normalized_z
            result[valid_mask] = self.normalized_z[valid_rows.astype(int), valid_cols.astype(int)]

            return result

            #return self.normalized_z[position[: , 0].astype(int), position[: , 1].astype(int)]

    def render(self):

        """ Show the ground truth """
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            
            self.ax_plot = self.ax.imshow(self.read(), cmap='inferno', interpolation='none')
        else:
            self.ax_plot.set_data(self.read())


        plt.draw()
        plt.pause(0.1)

    def step(self):
        """ Move every maximum with a random walk noise """

        self.A += self.dt*(2*(np.random.rand(*self.A.shape)-0.5) * self.xy_size * 0.9 + self.xy_size*0.1)
        self.create_field()

        pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ypacarai_map = np.genfromtxt('../Maps/map.txt',delimiter=' ',dtype=int)
    gt = shekel(ypacarai_map, max_number_of_peaks=6, is_bounded=True, seed=10, dt=0.01)

    
    
    for i in range(10):
        gt.reset()
        gt.render()

        #np.savetxt('test_DELETEME.csv', gt.read(), fmt='%f', delimiter=';')

        input()
        """
        for t in range(150):
            gt.step()
            gt.render()
        """





