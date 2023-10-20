import numpy as np
import numba

@numba.jit(nopython=True)
def set_seed(value):
    np.random.seed(value)

@numba.jit(nopython=True)
def gaussian_filter_2d(image, sigma):
    kernel_size = 5  # Asegurar un tamaño impar para el kernel
    radius = 2
    kernel = np.empty((kernel_size, kernel_size), dtype=np.float64)
    filtered_image = np.empty_like(image, dtype=np.float64)

    # Generar el kernel gaussiano
    sum_val = 0.0
    for y in range(kernel_size):
        for x in range(kernel_size):
            dx = x - radius
            dy = y - radius
            kernel[y, x] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            sum_val += kernel[y, x]

    # Normalizar el kernel
    kernel /= sum_val

    # Aplicar el filtro con evaluación en los bordes
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            acc = 0.0
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    px = x + kx - radius
                    py = y + ky - radius
                    if px >= 0 and py >= 0 and px < image.shape[1] and py < image.shape[0]:
                        acc += image[py, px] * kernel[ky, kx]
            filtered_image[y, x] = acc

    return filtered_image

@numba.jit(nopython=True)
def simulate_fire_front(M, N, fuel_map, max_steps):

    fireable_positions = np.argwhere(fuel_map == 1)
    n_cells = fireable_positions.shape[0]
    fire_array = np.zeros((n_cells, ), dtype=np.float64)
    ignition_array = np.zeros((n_cells, ))
    fuel_array = np.ones((n_cells, ))
    p_ignition_array = np.zeros((n_cells, ))

    temperature_map = np.zeros((M,N), dtype=np.float64)
    temperature_maps = np.zeros((max_steps,M,N))

    # Initial_point (random)
    initial_point = np.array((np.random.randint(0, n_cells),))
    fire_array[initial_point] = 1

    # Wind
    wind = np.array([np.random.uniform(0, 0.1)*0.0, np.random.uniform(0, 0.1)*0.0])

    # Ignition factor
    ignition_factor = 0.01

    # Max distance
    max_distance = 10

    # Fuel decay
    fuel_decay = 0.1

    for t in range(max_steps):

        # Compute the probability of ignition for each cell
        for idx, cell in enumerate(fireable_positions):

            if fire_array[idx] == 1:
                dx = cell[0] - fireable_positions[:,0]
                dy = cell[1] - fireable_positions[:,1]
                diferences = np.column_stack((dx, dy))
                wind_factor = np.array([np.sum(diference * wind) for diference in diferences])
                distances = np.array([np.sum(diference**2) + 0.01 for diference in diferences])
                p_ignition_array = ignition_factor * (1 + wind_factor) / distances**2

                ignition_array = ignition_array + p_ignition_array

                fuel_array[idx] = fuel_array[idx] - fuel_decay

                if fuel_array[idx] <= 0:
                    fire_array[idx] = 0
                    ignition_array[idx] = 0
                    fuel_array[idx] = 0

            
        # Compute the new fire map
        started_fire = np.array([np.random.rand() < ignition_array[idx] and fuel_array[idx] > 0.0 for idx in range(n_cells)])
        fire_array[started_fire] = 1.0
        

        # Compute the temperature map with the fire map
        for idx, fireable_position in enumerate(fireable_positions):
            temperatures_map[t, int(fireable_position[0]), int(fireable_position[1])] = fire_array[idx]


    
    return temperature_maps

class FireFrontGroundTruthNumba:

    def __init__(self, grid: np.ndarray, max_steps = 120, t0 = 0):

        self.grid = grid
        self.t = t0
        self.t0 = t0
        self.temperature_map = None
        self.max_steps = max_steps
        self.fig = None

    def reset(self):

        self.temperature_map = simulate_fire_front(self.grid.shape[0], self.grid.shape[1], self.grid, self.max_steps)
        self.t = self.t0

    def step(self):

        self.t += 1
        assert self.t < self.temperature_map.shape[0], "Simulation has ended. Increase max_steps."

    def read(self, position = None):
        
        if position is None:
            return self.temperature_map[self.t,:,:]
        else:
            return self.temperature_map[self.t, position[:,0].astype(int), position[:,1].astype(int)]

    def render(self):
        
        f_map = self.temperature_map[self.t]

        if self.fig is None:
            self.fig, self.ax = plt.subplots(1,1)
            self.d = self.ax.imshow(f_map, cmap = 'hot', vmin=0.0, vmax = 1.0)
            
            
        else:
            self.d.set_data(f_map)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time

    plt.ion()

    T = 100
    M = 58
    N = 60
    nav_map = np.genfromtxt("Environment\Maps\map.txt")

    set_seed(10)
    
    gt = FireFrontGroundTruthNumba(nav_map, 120)

    gt.reset()

    for _ in range(10):
        t0 = time.time()
        gt.reset()
        gt.render()

        for t in range(100):
            gt.step()
            gt.render()
        
        print(time.time() - t0)
        















    