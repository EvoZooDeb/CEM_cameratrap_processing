import numpy as np
from PytorchWildlife.models import classification as pw_classification
import cv2
import os
import pandas as pd

classification_model = pw_classification.DeepfauneClassifier() # Model weights are automatically downloaded.
prediction_path = "/directory/containing/detection/results/"
image_path      = "/directory/containing/images/"
result_path     = "/result/directory/"

# Class names and codes
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

# General class names and codes
gen_names = {
  0: "Animal",
  1: "Person",
  2: "Vehicle",
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
                    image_name = frame.split(".",)[0] + ".JPG"
                    image = cv2.imread(image_path + image_name)
                    h_image, w_image = image.shape[:2]
                    frame_count = frame_count + 1
                    frame_data  = []
                    group_size = 0
                    with open(path + frame) as f:
                        for line in f.readlines():
                            label, x_center, y_center, w_box, h_box = line.rstrip().split(" ")
                            
                            # If animal is in the bounding box
                            if float(label) != 1 and float(label) != 2:
                                
                                # Crop and resize bbox image
                                h_box_px = float(h_box) * float(h_image)
                                w_box_px = float(w_box) * float(w_image)
                                if h_box_px >= w_box_px:
                                    crop_size = h_box_px / 2
                                else:
                                    crop_size = w_box_px / 2
                                x_center_px = float(x_center) * float(w_image)
                                y_center_px = float(y_center) * float(h_image)
                                x_min = int(x_center_px - crop_size)
                                x_max = int(x_center_px + crop_size)
                                y_min = int(y_center_px - crop_size)
                                y_max = int(y_center_px + crop_size)
                                if x_min < 0:
                                    x_max = x_max - x_min
                                    x_min = 0
                                if x_max > w_image:
                                    x_min = x_min - (x_max - w_image)
                                    x_max = w_image
                                if y_min < 0:
                                    y_max = y_max - y_min
                                    y_min = 0
                                if y_max > h_image:
                                    y_min = y_min - (y_max - h_image)
                                    y_max = h_image

                                # Crop the animal
                                cropped_image = image[y_min:y_max, x_min:x_max]
                                h_cropped, w_cropped = cropped_image.shape[:2]
                                cropped_image = cv2.resize(cropped_image, (224, 224))
    
                                # Run deepfaune classification model
                                classification_results = classification_model.single_image_classification(cropped_image)
                                
                                # TOP1
                                #label = classification_results['prediction']
                                # Use scientific names
                                #if label == "badger":
                                #    label = "Meles_meles"
                                #elif label == "red deer":
                                #    label = "Cervus_elaphus"
                                #elif label == "cat":
                                #    label = "Felis_silvestris"
                                #elif label == "roe deer":
                                #    label = "Capreolus_capreolus"
                                #elif label == "dog":
                                #    label = "Dog"
                                #elif label == "wolf":
                                #    label = "Canis_lupus"
                                #elif label == "mouflon":
                                #    label = "Ovis_orientalis"
                                #elif label == "fox":
                                #    label = "Vulpes_vulpes"
                                #elif label == "wild boar":
                                #    label = "Sus_scrofa"
                                #else:
                                #    label = "Animal"

                                # TOP3
                                species = []
                                confidences = []
                                for i in classification_results['all_confidences']:
                                    species.append(i[0])
                                    confidences.append(i[1])
                                confidences = np.array([confidences])
                                top3_idx    = np.argpartition(confidences[0], -3)[-3:]
                                label = species[top3_idx[0]] + "/" + species[top3_idx[1]] + "/" + species[top3_idx[2]]
                                label = label.replace("badger", "Meles_meles")
                                label = label.replace("red deer", "Cervus_elaphus")
                                label = label.replace("cat", "Felis_silvestris")
                                label = label.replace("roe deer", "Capreolus_capreolus")
                                label = label.replace("dog", "Dog")
                                label = label.replace("wolf", "Canis_lupus")
                                label = label.replace("mouflon", "Ovis_orientalis")
                                label = label.replace("fox", "Vulpes_vulpes")
                                label = label.replace("wild boar", "Sus_scrofa")

                                frame_data.append([str(label)])
                            else:
                                frame_data.append([f"{gen_names.get(int(label))}"])
                            group_size = group_size + 1
                    print(frame_data)
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

prediction_results  = analyze_sequence(prediction_path, result_path + "DeepFaune_seq_class_TOP3_predicted_results.csv")
print(prediction_results)


