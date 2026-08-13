# Load libraries
import os 
import pandas as pd

# Define paths
prediction_path = "/directory/containing/detection/results/"
label_path      = "/directory/containing/groundtruth/results/"
result_path     = "/result/directory/"

# Class names and codes
#names = {
#  0: "Canis_lupus",
#  1: "Person",
#  2: "Cervus_elaphus",
#  3: "Capreolus_capreolus",
#  4: "Sus_scrofa",
#  5: "Vulpes_vulpes",
#  6: "Felis_silvestris",
#  7: "Meles_meles",
#  8: "Ovis_orientalis",
#  9: "Vehicle",
#  10: "Dog",
#}

# General class names and codes
#names = {
#  0: "Animal",
#  1: "Person",
#  2: "Vehicle",
#}

# Grouped class names and codes
names = {
  0: "Artiodactyla",
  1: "Person",
  2: "Vehicle",
  3: "Carnivora",
}

def analyze_frames(path, result_path):
    file_names   = os.listdir(path)
    frame_names  = [file for file in file_names if "frame" in file] # Keep only videos
    results      = []
    
    # For each file in directory
    for frame in frame_names:
        print("Working with frame: ", frame, "--------------------------------------------------------------------------------------")
        frame_data = []
        group_size = 0
        with open(path + frame) as f:
            for line in f.readlines():
                label, x_center, y_center, width, height = line.rstrip().split(" ")
                frame_data.append([f"{names.get(int(label))}"])
                group_size = group_size + 1
        frame_df = pd.DataFrame(frame_data, columns = ['label'])
        most_frequent_label = frame_df['label'].value_counts()[:1].index.tolist()[0]
        most_frequent_label_count = frame_df[frame_df['label'] == most_frequent_label].count().iloc[0]
        confidence = most_frequent_label_count / group_size
        results.append([frame, most_frequent_label, confidence, group_size])
    result_df = pd.DataFrame(results, columns = ['file_name', 'label', 'label_confidence', 'group_size'])
    result_df.to_csv(result_path, sep = ",", encoding = "utf-8")
    return result_df

#ground_truth_results = analyze_frames(label_path, result_path + "grp_frame_ground_truth_results.csv")
#print(ground_truth_results)
prediction_results  = analyze_frames(prediction_path, result_path + "train28_frame_predicted_results.csv")
print(prediction_results)
