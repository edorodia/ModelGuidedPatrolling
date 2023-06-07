from torch.utils.data import Dataset
import torch as th
import numpy as np

class StaticDataset(Dataset):

	""" Dataset class for the static environment """

	def __init__(self, path_trajectories, path_gts, transform=None):

		

		self.trajectories = np.load(path_trajectories)
		self.trajectories = self.trajectories.reshape(self.trajectories.shape[0]*self.trajectories.shape[1], self.trajectories.shape[2], self.trajectories.shape[3], self.trajectories.shape[4])
		self.gts = np.load(path_gts)
		self.transform = transform


	def __len__(self):
		
		# The length of the dataset is the number of trajectories times the number of frames in each trajectory
		return len(self.trajectories)

	def __getitem__(self, idx):
		
		# Transform the index to a list
		if th.is_tensor(idx):
			idx = idx.tolist()

		# Get the trajectory of idx
		observation = self.trajectories[idx]

		# Get the ground truth that corresponds to the trajectory
		gt = self.gts[idx // (len(self.trajectories) // len(self.gts))]

		sample = (observation, gt)

		if self.transform:
			sample = (self.transform(sample), self.transform(gt))

		return sample


if __name__ == '__main__':

    # Load the dataset and plot it #

    import matplotlib.pyplot as plt

    dataset = StaticDataset(path_trajectories = 'ModelTrain/Data/trajectories_static.npy',
                            path_gts = 'ModelTrain/Data/gts_static.npy',

                            transform=None)

    mask = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')

    print("Shape of the dataset: ", dataset.trajectories.shape)

    # Show some samples in a 3x3 grid

    fig, axs = plt.subplots(3, 3, figsize = (10, 10))

    for i in range(3):

            j = np.random.randint(0, 1000)
            h = np.random.randint(0, 10)

            axs[i, 0].imshow(dataset[j*h][0][0,:,:], vmin=0, vmax=1)
            axs[i, 1].imshow(dataset[j*h][0][1,:,:], vmin=0, vmax=1)
            axs[i, 2].imshow(dataset[j*h][1], vmin=0, vmax=1)
            axs[i, 0].set_title(f'Time traj {j} step {h}')
            axs[i, 1].set_title(f'Model traj {j} step {h}')
            axs[i, 2].set_title(f'GT traj {j} step {h}')

    plt.show()