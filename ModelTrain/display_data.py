import numpy as np

loaded_data = np.load('ModelTrain\Data\gts_shekel_train.npy')
print("Ground Truths saved -> length: " + str(loaded_data.shape[0]))
#print(loaded_data)

loaded_data = np.load('ModelTrain\Data\\trajectories_shekel_train.npy')
print("Trajectories saved -> length: " + str(loaded_data.shape[1]))
#print(loaded_data)



