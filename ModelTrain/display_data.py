import numpy as np

folder = "NoNoise_-_False_False_none"

loaded_data = np.load('ModelTrain/NoNoise_Shekel_Datasets/' + folder + '/gts_shekel_train.npy', mmap_mode="r")
print("Ground Truths saved -> length: " + str(loaded_data.shape))
#print(loaded_data)

loaded_data = np.load('ModelTrain/NoNoise_Shekel_Datasets/' + folder + '/trajectories_shekel_train.npy', mmap_mode="r")
print("Trajectories saved -> length: " + str(loaded_data.shape))
#print(loaded_data)



