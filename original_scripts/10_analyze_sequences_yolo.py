# Load libraries
import os 
import pandas as pd

prediction_path = "/home/golah/wolf_camtrap/background_test/NonE_train28_25/labels/"
#label_path      = "/home/golah/wolf_camtrap/training_2023/yolo_annotations_test_4_combined/"
label_path      = "/home/golah/wolf_camtrap/training_2023/test/labels/"
result_path     = "/home/golah/wolf_camtrap/sequence_analysis/"
names = {
  0: "Canis_lupus",
  1: "Person",
  2: "Cervus_elaphus",
  3: "Capreolus_capreolus",
  4: "Sus_scrofa",
  5: "Vulpes_vulpes",
  6: "Felis_silvestris",
  7: "Meles_meles",
  8: "Ovis_orientalis",
  9: "Vehicle",
  10: "Dog",
}

# Combined
#gen_names = {
#  0: "Animal",
#  1: "Person",
#  2: "Vehicle",
#}

# Grouped
gen_names = {
  0: "Artiodactyla",
  1: "Person",
  2: "Vehicle",
  3: "Carnivora",
}

def analyze_sequence(path, result_path):
    file_names     = os.listdir(path)
    file_names     = [file for file in file_names if "frame" in file] # Keep only videos
    file_indexes   = [indexes.split("_",)[0] + "_" + indexes.split("_",)[1] for indexes in file_names]
    unique_indexes = pd.Series(file_indexes).drop_duplicates().tolist()
    results        = []
    # For each file in directory
    for index in unique_indexes:
        print("Working with index: ", index, "--------------------------------------------------------------------------------------")
        for id, species in names.items():
            frames = [i for i in file_names if i.startswith(index + "_") and i.endswith(species + ".txt")]
            if len(frames)>0:
                file_name = index + "_" + species
                file_data   = []
                frame_count = 0
                for frame in frames:
                    frame_count = frame_count + 1
                    frame_data  = []
                    group_size = 0
                    with open(path + frame) as f:
                        for line in f.readlines():
                            label, x_center, y_center, width, height = line.rstrip().split(" ")
                            frame_data.append([f"{gen_names.get(int(label))}"])
                            group_size = group_size + 1
                    frame_df = pd.DataFrame(frame_data, columns = ['label'])
                    most_frequent_label = frame_df['label'].value_counts()[:1].index.tolist()[0]
                    most_frequent_label_count = frame_df[frame_df['label'] == most_frequent_label].count().iloc[0]
                    number_of_annotated_objects = frame_df.count().iloc[0]
                    confidence = most_frequent_label_count / number_of_annotated_objects
                    file_data.append([file_name, most_frequent_label, confidence, number_of_annotated_objects, group_size])
                    #file_data.append([file_name, f"{names.get(int(label))}", confidence, number_of_annotated_objects, group_size])
                file_df = pd.DataFrame(file_data, columns = ['file_name', 'label', 'label_confidence','n_annotated_objects', 'group_size'])
                most_frequent_label = file_df['label'].value_counts()[:1].index.tolist()[0]
                most_frequent_label_count = file_df[file_df['label'] == most_frequent_label].count().iloc[0]
                #number_of_annotated_objects = file_df.count().iloc[0]
                number_of_annotated_objects = sum(file_df['n_annotated_objects'])
                number_of_annotated_frames = frame_count
                confidence = most_frequent_label_count / number_of_annotated_frames
                mean_group_size = sum(file_df['group_size'])/len(file_df['group_size'])
                max_group_size  = max(file_df['group_size'])
                #results.append([file_name, f"{names.get(int(label))}", confidence, str(number_of_annotated_objects) ,str(number_of_annotated_frames), mean_group_size, max_group_size])
                results.append([file_name, most_frequent_label, confidence, str(number_of_annotated_objects) ,str(number_of_annotated_frames), mean_group_size, max_group_size])
    result_df = pd.DataFrame(results, columns = ['file_name', 'label', 'label_confidence','n_annotated_objects' ,'n_annotated_frames', 'mean_group_size', 'max_group_size'])
    result_df.to_csv(result_path, sep = ",", encoding = "utf-8")
    return result_df

ground_truth_results = analyze_sequence(label_path, result_path + "grp_seq_ground_truth_results.csv")
print(ground_truth_results)
prediction_results  = analyze_sequence(prediction_path, result_path + "train28_seq_predicted_results.csv")
print(prediction_results)
