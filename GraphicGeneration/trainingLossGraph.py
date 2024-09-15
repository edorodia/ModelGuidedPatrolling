import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

run_names = {
    2 : 'NoNoise_-_False_False_none',
    3 : 'NoNoise_-_True_False_none',
    4 : 'MeanNoise_-_False_False_none',
    5 : 'MeanNoise_-_True_False_none',
    6 : 'FishEyeNoise-_False_False_none',
    7 : 'FishEyeNoise-_True_False_none',
    8 : 'NoDrone_-_False_none',

}

matplotlib.use('Agg')

for runs_id in np.arange(2, 9):
    # Define the path to your JSON file
    json_file_path = 'TrainData/json' + str(runs_id) + '.json'

    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Convert JSON data to a NumPy array
    # Assuming the JSON data is a list of lists or a list of arrays
    np_array = np.array(data)

    np_array[:, 1] = np_array[:, 1].astype(int)

    title = "Run_" + run_names[runs_id]

    width = 40
    height = 25

    linewidth_global = 5

    titlesize = 55
    labelsize = 40
    axissize = 35
    legendsize = 40

    background_color = '#eaeaf2'

    save_graph = True

    plt.figure(figsize=(width,height))
    plt.gca().set_facecolor(background_color)
    plt.title(title, fontsize=titlesize)
    plt.xlabel("Epoch", fontsize=labelsize)
    plt.ylabel("Loss", fontsize=labelsize)
    plt.plot(np_array[:, 1], np_array[:, 2], color = 'blue', linestyle='-', linewidth=linewidth_global)
    plt.tick_params(axis='both', which='major', labelsize=axissize)
    if save_graph:
        plt.savefig('runs_' + str(runs_id) + '_' + run_names[runs_id] + '.pdf', format='pdf')