import numpy as np

loaded_data = np.load('ModelTrain\Data\gts_shekel_train.npy', mmap_mode="r")
print("Ground Truths saved -> length: " + str(loaded_data.shape))
print(loaded_data)

loaded_data = np.load('ModelTrain\Data\\trajectories_shekel_train.npy', mmap_mode="r")
print("Trajectories saved -> length: " + str(loaded_data.shape))
#print(loaded_data)



