# Load libraries
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ground_truth_path = "/home/golah/wolf_camtrap/sequence_analysis/gen_seq_ground_truth_results.csv"
prediction_path   = "/home/golah/wolf_camtrap/sequence_analysis/DeepFaune_seq_predicted_results.csv"
labels = ["Canis_lupus", "Person", "Cervus_elaphus", "Capreolus_capreolus", "Sus_scrofa", "Vulpes_vulpes", "Felis_silvestris", "Meles_meles", "Ovis_orientalis", "Vehicle", "Dog"]
pred       = pd.read_csv(prediction_path)
truth      = pd.read_csv(ground_truth_path)
results    = []
mean_error = []
max_error  = []
for i in range(0, truth.shape[0], 1):
    t_row = truth.iloc[i]
    #if t_row['n_annotated_frames'] > 1 and t_row['label'] == labels[0]:
    if t_row['n_annotated_frames'] > 1:
        p_row = pred[pred['file_name'] == t_row['file_name']]
        if len(p_row) > 0:
            if t_row['label'] == p_row['label'].values[0]:
                match = 1
            else:
                match = 0
            results.append(match)
            mean_diff = t_row['mean_group_size'] - p_row['mean_group_size'].values[0]
            mean_error.append(mean_diff)
            max_diff  = t_row['max_group_size'] - p_row['max_group_size'].values[0]
            max_error.append(max_diff)
        else:
            results.append(0)
            mean_error.append(t_row['mean_group_size'])
            max_error.append(t_row['max_group_size'])


# Calculate and visualize count errors
# Mean group size error
mean_error = np.array(mean_error)
print("Mean group error is 0: ", round(len(mean_error[np.where(mean_error == 0)]) / len(mean_error), 3))
print("Mean group error is -0.5:0.5: ", round(len(mean_error[np.where((mean_error >= -0.50) & (mean_error <= 0.50))]) / len(mean_error), 3))

#labels, counts = np.unique(np.array(mean_error), return_counts = True)
#print(labels, counts, len(max_error))
#bins = np.linspace(-4, 4, 17)
#plt.hist(np.array(mean_error), bins = bins)
#plt.xticks(bins)
#plt.show()

# Max group size error
max_error = np.array(max_error)
labels, counts = np.unique(max_error, return_counts = True)
print("Max group size error is 0: ", round(len(max_error[np.where(max_error == 0)]) / len(max_error), 3))
print("Max group size error is -1:1: ", round(len(max_error[np.where((max_error == -1) | (max_error == 0) | (max_error == 1))])  / len(max_error), 3))
#plt.bar(labels, counts, align='center')
#plt.gca().set_xticks(labels)
#plt.show()
#print(results)
#print(sum(results))
#print(len(results))
print("Classification accuracy: ", round(sum(results) / len(results), 3))
