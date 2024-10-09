import numpy as np 
import matplotlib.pyplot as plt
import math
import pandas as pd

peak_existent = False
peak_threshold = 300
group_dim = 30


def smoother(original):
    data = pd.Series(original)
    smoothed = data.rolling(window = group_dim, step = group_dim).mean()
    return smoothed

def smoother_with_peak(original):
    count = 0
    global_count = 0
    accumulator = 0
    smoothed = []
    smoothed_step = []
    for ele in original:
        if global_count % peak_threshold == 0:
            #add the peak value
            smoothed.append(ele)
            smoothed_step.append(global_count)
            global_count += 1
            accumulator = 0
            count = 0
        else:
            accumulator += ele
            count += 1
            global_count += 1
            if count == group_dim:
                smoothed.append(accumulator / group_dim)
                smoothed_step.append(global_count)
                count = 0
                accumulator = 0
    return [smoothed_step, smoothed]

model1 = "NoNoise"
model2 = "FishEyeNoise"
model3 = "MeanNoise"
model4 = "NoDrone"

test_name = "_Dynamic_Peaks_Variation_long_run"

step_list_model1 = np.load("GraphData/step_list_"+ model1 + test_name +".npy")
step_mse_model1 = np.load("GraphData/step_mse_"+ model1 + test_name +".npy")
step_rmse_model1 = np.load("GraphData/step_rmse_"+ model1 + test_name +".npy")
step_w_rmse_model1 = np.load("GraphData/step_w_rmse_"+ model1 + test_name +".npy")

step_list_model2 = np.load("GraphData/step_list_"+ model2 + test_name +".npy")
step_mse_model2 = np.load("GraphData/step_mse_"+ model2 + test_name +".npy")
step_rmse_model2 = np.load("GraphData/step_rmse_"+ model2 + test_name +".npy")
step_w_rmse_model2 = np.load("GraphData/step_w_rmse_"+ model2 + test_name +".npy")

step_list_model3 = np.load("GraphData/step_list_"+ model3 + test_name +".npy")
step_mse_model3 = np.load("GraphData/step_mse_"+ model3 + test_name +".npy")
step_rmse_model3 = np.load("GraphData/step_rmse_"+ model3 + test_name +".npy")
step_w_rmse_model3 = np.load("GraphData/step_w_rmse_"+ model3 + test_name +".npy")

step_list_model4 = np.load("GraphData/step_list_"+ model4 + test_name +".npy")
step_mse_model4 = np.load("GraphData/step_mse_"+ model4 + test_name +".npy")
step_rmse_model4 = np.load("GraphData/step_rmse_"+ model4 + test_name +".npy")
step_w_rmse_model4 = np.load("GraphData/step_w_rmse_"+ model4 + test_name +".npy")


if peak_existent == False:
    ### Model 1 ###
    step_dimension = len(step_mse_model1)
    new_step_dimension = math.ceil(step_dimension / group_dim)
    step_list_model1_new = np.arange(0, new_step_dimension)

    step_list_model1_new = step_list_model1_new * group_dim
    s_step_mse_model1 = smoother(step_mse_model1)
    s_step_mse_model1[0] = step_mse_model1[0]
    s_step_rmse_model1 = smoother(step_rmse_model1)
    s_step_rmse_model1[0] = step_rmse_model1[0]
    s_step_w_rmse_model1 = smoother(step_w_rmse_model1)
    s_step_w_rmse_model1[0] = step_w_rmse_model1[0]


    ### Model 2 ###
    step_dimension = len(step_mse_model2)
    new_step_dimension = math.ceil(step_dimension / group_dim)
    step_list_model2_new = np.arange(0, new_step_dimension)

    step_list_model2_new = step_list_model2_new * group_dim
    s_step_mse_model2 = smoother(step_mse_model2)
    s_step_mse_model2[0] = step_mse_model2[0]
    s_step_rmse_model2 = smoother(step_rmse_model2)
    s_step_rmse_model2[0] = step_rmse_model2[0]
    s_step_w_rmse_model2 = smoother(step_w_rmse_model2)
    s_step_w_rmse_model2[0] = step_w_rmse_model2[0]


    ### Model 3 ###
    step_dimension = len(step_mse_model3)
    new_step_dimension = math.ceil(step_dimension / group_dim)
    step_list_model3_new = np.arange(0, new_step_dimension)

    step_list_model3_new = step_list_model3_new * group_dim
    s_step_mse_model3 = smoother(step_mse_model3)
    s_step_mse_model3[0] = step_mse_model3[0]
    s_step_rmse_model3 = smoother(step_rmse_model3)
    s_step_rmse_model3[0] = step_rmse_model3[0]
    s_step_w_rmse_model3 = smoother(step_w_rmse_model3)
    s_step_w_rmse_model3[0] = step_w_rmse_model3[0]


    ### Model 4 ###
    step_dimension = len(step_mse_model4)
    new_step_dimension = math.ceil(step_dimension / group_dim)
    step_list_model4_new = np.arange(0, new_step_dimension)

    step_list_model4_new = step_list_model4_new * group_dim
    s_step_mse_model4 = smoother(step_mse_model4)
    s_step_mse_model4[0] = step_mse_model4[0]
    s_step_rmse_model4 = smoother(step_rmse_model4)
    s_step_rmse_model4[0] = step_rmse_model4[0]
    s_step_w_rmse_model4 = smoother(step_w_rmse_model4)
    s_step_w_rmse_model4[0] = step_w_rmse_model4[0]

elif peak_existent:
    ### Model 1 ###
    [step_list_model1_new, s_step_mse_model1] = smoother_with_peak(step_mse_model1)
    s_step_rmse_model1 = smoother_with_peak(step_rmse_model1)[1]
    s_step_w_rmse_model1 = smoother_with_peak(step_w_rmse_model1)[1]


    ### Model 2 ###
    [step_list_model2_new, s_step_mse_model2] = smoother_with_peak(step_mse_model2)
    s_step_rmse_model2 = smoother_with_peak(step_rmse_model2)[1]
    s_step_w_rmse_model2 = smoother_with_peak(step_w_rmse_model2)[1]


    ### Model 3 ###
    [step_list_model3_new, s_step_mse_model3] = smoother_with_peak(step_mse_model3)
    s_step_rmse_model3 = smoother_with_peak(step_rmse_model3)[1]
    s_step_w_rmse_model3 = smoother_with_peak(step_w_rmse_model3)[1]


    ### Model 4 ###
    [step_list_model4_new, s_step_mse_model4] = smoother_with_peak(step_mse_model4)
    s_step_rmse_model4 = smoother_with_peak(step_rmse_model4)[1]
    s_step_w_rmse_model4 = smoother_with_peak(step_w_rmse_model4)[1]

print(step_list_model1_new)
print(len(step_list_model1_new))
print(len(s_step_mse_model1))

np.save('GraphData/step_list_'+ model1 + test_name + '_smoothed_' + str(group_dim) + '.npy', step_list_model1_new)
np.save('GraphData/step_mse_'+ model1 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_mse_model1)
np.save('GraphData/step_rmse_'+ model1 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_rmse_model1)
np.save('GraphData/step_w_rmse_'+ model1 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_w_rmse_model1)

np.save('GraphData/step_list_'+ model2 + test_name + '_smoothed_' + str(group_dim) + '.npy', step_list_model2_new)
np.save('GraphData/step_mse_'+ model2 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_mse_model2)
np.save('GraphData/step_rmse_'+ model2 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_rmse_model2)
np.save('GraphData/step_w_rmse_'+ model2 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_w_rmse_model2)

np.save('GraphData/step_list_'+ model3 + test_name + '_smoothed_' + str(group_dim) + '.npy', step_list_model3_new)
np.save('GraphData/step_mse_'+ model3 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_mse_model3)
np.save('GraphData/step_rmse_'+ model3 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_rmse_model3)
np.save('GraphData/step_w_rmse_'+ model3 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_w_rmse_model3)

np.save('GraphData/step_list_'+ model4 + test_name + '_smoothed_' + str(group_dim) + '.npy', step_list_model4_new)
np.save('GraphData/step_mse_'+ model4 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_mse_model4)
np.save('GraphData/step_rmse_'+ model4 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_rmse_model4)
np.save('GraphData/step_w_rmse_'+ model4 + test_name + '_smoothed_' + str(group_dim) + '.npy', s_step_w_rmse_model4)




