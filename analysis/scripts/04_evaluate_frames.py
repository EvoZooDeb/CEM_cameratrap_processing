# Load libraries
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define path
ground_truth_path = "/directory/containing/groundtruth/results/ss_frame_ground_truth_results.csv"
prediction_path   = "/directory/containing/classification/results//DeepFaune_frame_class_TOP3_predicted_results.csv"
labels = ["Canis_lupus", "Person", "Cervus_elaphus", "Capreolus_capreolus", "Sus_scrofa", "Vulpes_vulpes", "Felis_silvestris", "Meles_meles", "Ovis_orientalis", "Vehicle", "Dog"]
pred       = pd.read_csv(prediction_path)
truth      = pd.read_csv(ground_truth_path)
results    = []
group_error  = []
for i in range(0, truth.shape[0], 1):
    t_row = truth.iloc[i]
    p_row = pred[pred['file_name'] == t_row['file_name']]
    if len(p_row) > 0:
        #if t_row['label'] == p_row['label'].values[0]: # TOP1
        if t_row['label'] in p_row['label'].values[0]:  # TOP3
            match = 1
        else:
            match = 0
        results.append(match)
        group_diff = t_row['group_size'] - p_row['group_size'].values[0]
        group_error.append(group_diff)
    else:
        results.append(0)
        group_error.append(t_row['group_size'])

# Calculate and visualize count errors
# Group size error
group_error = np.array(group_error)
labels, counts = np.unique(group_error, return_counts = True)
print("Max group size error is 0: ", round(len(group_error[np.where(group_error == 0)]) / len(group_error), 3))
print("Max group size error is -1:1: ", round(len(group_error[np.where((group_error == -1) | (group_error == 0) | (group_error == 1))])  / len(group_error), 3))
#plt.bar(labels, counts, align='center')
#plt.gca().set_xticks(labels)
#plt.show()
#print(results)
#print(sum(results))
#print(len(results))
print("Classification accuracy: ", round(sum(results) / len(results), 3))
