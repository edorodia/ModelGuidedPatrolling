import numpy as np 
import matplotlib.pyplot as plt

first_model = "NoDrone"
second_model = "NoNoise"
comparison_name = first_model + "-" + second_model

step_list_NoDrone = np.load("step_list_"+ first_model +".npy")
step_mse_NoDrone = np.load("step_mse_"+ first_model +".npy")
step_rmse_NoDrone = np.load("step_rmse_"+ first_model +".npy")
step_w_rmse_NoDrone = np.load("step_w_rmse_"+ first_model +".npy")

step_list_FishEyeNoise = np.load("step_list_"+ second_model +".npy")
step_mse_FishEyeNoise = np.load("step_mse_"+ second_model +".npy")
step_rmse_FishEyeNoise = np.load("step_rmse_"+ second_model +".npy")
step_w_rmse_FishEyeNoise = np.load("step_w_rmse_"+ second_model +".npy")


plt.figure(figsize=(10,6))
plt.title('MSE comparison')
plt.plot(step_list_NoDrone, step_mse_NoDrone, label= first_model + ' - MSE', color = 'blue', linestyle='-')
plt.plot(step_list_FishEyeNoise, step_mse_FishEyeNoise, label= second_model + ' - MSE', color = 'green', linestyle='-')
plt.legend()
plt.savefig(comparison_name + '_Simulation_Comparison_MSE.pdf', format='pdf')

plt.figure(figsize=(10,6))
plt.title('RMSE comparison')
plt.plot(step_list_NoDrone, step_rmse_NoDrone, label= first_model + ' - RMSE', color = 'blue', linestyle='-')
plt.plot(step_list_FishEyeNoise, step_rmse_FishEyeNoise, label= second_model + ' - RMSE', color = 'green', linestyle='-')
plt.legend()
plt.savefig(comparison_name + '_Simulation_Comparison_RMSE.pdf', format='pdf')

plt.figure(figsize=(10,6))
plt.title('W_RMSE comparison')
plt.plot(step_list_NoDrone, step_w_rmse_NoDrone, label= first_model + ' - W_RMSE', color = 'blue', linestyle='-')
plt.plot(step_list_FishEyeNoise, step_w_rmse_FishEyeNoise, label= second_model + ' - W_RMSE', color = 'green', linestyle='-')
plt.legend()
plt.savefig(comparison_name + '_Simulation_Comparison_W_RMSE.pdf', format='pdf')

plt.show()








