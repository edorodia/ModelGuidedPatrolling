# Compare the Models results
import sys
sys.path.append('.')
import numpy as np
import argparse
from typing import List

parser = argparse.ArgumentParser(description='Compare_Models')

parser.add_argument('--benchmark', type=str, default='shekel', choices=['shekel', 'algae_bloom'])
parser.add_argument('--test_type', type=str,  choices=['test', 'validation'])

args = parser.parse_args()

#################### Folders and Parameters setting ####################

# Prefix for the type of test and benchmark used
test_benchmark_prefix = args.test_type + "_" + args.benchmark +  "_"

# Name of the models that have been tested and have their file of results
models_to_compare = [
    "NoNoise_-_False_False_none",
    "NoNoise_-_True_False_none"
]
#######################################################################

#minimum values
min_mse = min_rmse = min_w_rmse = min_sec_per_op = max_r2 = None

#where was the minimum value found
min_mse_model = min_rmse_model = min_w_rmse_model = min_sec_per_op_model = max_r2_model = None

file_names: List[str] = []

for model in models_to_compare:
    file_names.append(test_benchmark_prefix + model + ".txt")

for file_name in file_names:
    with open(file_name, 'r') as file:
        lines = file.readlines()
    
    model = file_name.replace(test_benchmark_prefix, "")

    for line in lines:
        words = line.split()
        if      'MSE' == words[0]:
            mse = float(words[2])
        elif    'RMSE' == words[0]:
            rmse = float(words[2])
        elif    'W_RMSE' == words[0]:
            w_rmse = float(words[2])
        elif    'SEC_PER_OP' == words[0]:
            sec_per_op = float(words[2])
        elif    'R2' == words[0]:
            r2 = float(words[2])
        
    if all(var is None for var in [min_mse, min_rmse, min_w_rmse, min_sec_per_op, max_r2]):
        min_mse = mse
        min_rmse = rmse
        min_w_rmse = w_rmse
        min_sec_per_op = sec_per_op
        max_r2 = r2
        min_mse_model = min_rmse_model = min_w_rmse_model = min_sec_per_op_model = max_r2_model = model
    else:
        if mse < min_mse:
            min_mse = mse
            min_mse_model = model
        if rmse < min_rmse:
            min_rmse = rmse
            min_rmse_model = model
        if w_rmse < min_w_rmse:
            min_w_rmse = w_rmse
            min_w_rmse_model = model
        if sec_per_op < min_sec_per_op:
            min_sec_per_op = sec_per_op
            min_sec_per_op_model = model
        if r2 > max_r2:
            max_r2 = r2
            max_r2_model = model

name_comparison_file = "comparison"
for model in models_to_compare:
    name_comparison_file += "_" + model

with open(name_comparison_file + ".txt", 'w') as file:
    file.write(f"{min_mse_model:<30} - {'MSE':<10} -: {min_mse:.20f}\n")
    file.write(f"{min_rmse_model:<30} - {'RMSE':<10} -: {min_rmse:.20f}\n")
    file.write(f"{min_w_rmse_model:<30} - {'W_RMSE':<10} -: {min_w_rmse:.20f}\n")
    file.write(f"{min_sec_per_op_model:<30} - {'SEC_PER_OP':<10} -: {min_sec_per_op:.20f}\n")
    file.write(f"{max_r2_model:<30} - {'R2':<10} -: {max_r2:.20f}\n")

