import numpy as np

benchmark = "train"

folder = "FishEyeNoise_-_True_False_none"

loaded_data = np.load('ModelTrain/'+ 'Training_FishEyeNoise_Shekel_Datasets/' + folder + '/gts_shekel_'+ benchmark +'.npy', mmap_mode="r")
print("Ground Truths saved -> length: " + str(loaded_data.shape))
#print(loaded_data)

loaded_data = np.load('ModelTrain/'+ 'Training_FishEyeNoise_Shekel_Datasets/' + folder + '/trajectories_shekel_'+ benchmark +'.npy', mmap_mode="r")
print("Trajectories saved -> length: " + str(loaded_data.shape))
#print(loaded_data)



