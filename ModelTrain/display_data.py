import numpy as np

benchmark = "test"

folder = "test_FishEyeNoise_Shekel_Datasets/FishEyeNoise_-_True_False_none"
#folder = 'ModelTrain/Data'

#loaded_data = np.load('ModelTrain/'+ 'test_FishEyeNoise_Shekel_Datasets_low_coverage/' + folder + '/gts_shekel_'+ benchmark +'_FishEyeNoise_NoNoise.npy', mmap_mode="r")
loaded_data = np.load(folder + '/gts_shekel_' + benchmark + '.npy', mmap_mode="r")
print("Ground Truths saved -> length: " + str(loaded_data.shape))
#print(loaded_data)

#loaded_data = np.load('ModelTrain/'+ 'test_FishEyeNoise_Shekel_Datasets_low_coverage/' + folder + '/trajectories_shekel_'+ benchmark +'_FishEyeNoise_NoNoise.npy', mmap_mode="r")
loaded_data = np.load(folder + '/trajectories_shekel_' + benchmark + '.npy', mmap_mode="r")
print("Trajectories saved -> length: " + str(loaded_data.shape))
#print(loaded_data)



