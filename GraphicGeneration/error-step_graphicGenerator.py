import numpy as np 
import matplotlib.pyplot as plt

model1 = "NoNoise"
model2 = "FishEyeNoise"
model3 = "MeanNoise"
model4 = "NoDrone"

comparison_name = "Static"

test_name = ""

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

width = 80
height = 170

linewidth_global = 40

titlesize = 250
labelsize = 250
axissize = 200
legendsize = 250

background_color = '#eaeaf2'

save_graph = True


plt.figure(figsize=(width,height))
# Imposta il formato dei tick con 2 decimali
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
plt.gca().set_facecolor(background_color)
plt.subplots_adjust(left=0.2) 
plt.title(comparison_name + ' - MSE comparison', fontsize=titlesize, pad = 100)
plt.xlabel("Step", fontsize=labelsize, labelpad=100)
plt.ylabel("MSE", fontsize=labelsize, labelpad=100)
plt.plot(step_list_model1, step_mse_model1, label= model1 + ' - MSE', color = 'blue', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model2, step_mse_model2, label= model2 + ' - MSE', color = 'green', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model3, step_mse_model3, label= model3 + ' - MSE', color = 'red', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model4, step_mse_model4, label= model4 + ' - MSE', color = 'purple', linestyle='-', linewidth=linewidth_global)
plt.legend(fontsize=legendsize)
plt.tick_params(axis='both', which='major', labelsize=axissize, pad = 100)
if save_graph:
    plt.savefig(comparison_name + '_Comparison_MSE.pdf', format='pdf')

plt.figure(figsize=(width,height))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
plt.subplots_adjust(left=0.2) 
plt.gca().set_facecolor(background_color)
plt.title(comparison_name + ' - RMSE comparison', fontsize=titlesize, pad = 150)
plt.xlabel("Step", fontsize=labelsize, labelpad=100)
plt.ylabel("RMSE", fontsize=labelsize, labelpad=100)
plt.plot(step_list_model1, step_rmse_model1, label= model1 + ' - RMSE', color = 'blue', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model2, step_rmse_model2, label= model2 + ' - RMSE', color = 'green', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model3, step_rmse_model3, label= model3 + ' - RMSE', color = 'red', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model4, step_rmse_model4, label= model4 + ' - RMSE', color = 'purple', linestyle='-', linewidth=linewidth_global)
plt.legend(fontsize=legendsize)
plt.tick_params(axis='both', which='major', labelsize=axissize, pad = 150)
if save_graph:
    plt.savefig(comparison_name + '_Comparison_RMSE.pdf', format='pdf')

plt.figure(figsize=(width,height))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
plt.subplots_adjust(left=0.2) 
plt.gca().set_facecolor(background_color)
plt.title(comparison_name + ' - W_RMSE comparison', fontsize=titlesize, pad = 100)
plt.xlabel("Step", fontsize=labelsize, labelpad=100)
plt.ylabel("W_RMSE", fontsize=labelsize, labelpad=100)
plt.plot(step_list_model1, step_w_rmse_model1, label= model1 + ' - W_RMSE', color = 'blue', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model2, step_w_rmse_model2, label= model2 + ' - W_RMSE', color = 'green', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model3, step_w_rmse_model3, label= model3 + ' - W_RMSE', color = 'red', linestyle='-', linewidth=linewidth_global)
plt.plot(step_list_model4, step_w_rmse_model4, label= model4 + ' - W_RMSE', color = 'purple', linestyle='-', linewidth=linewidth_global)
plt.legend(fontsize=legendsize)
plt.tick_params(axis='both', which='major', labelsize=axissize, pad = 100)
if save_graph:
    plt.savefig(comparison_name + '_Comparison_W_RMSE.pdf', format='pdf')

#plt.show()








