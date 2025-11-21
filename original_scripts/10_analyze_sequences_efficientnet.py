import numpy as np
import cv2
import os
import pandas as pd
import tensorflow as tf
from keras.utils import img_to_array

# Species model
#model_path = "/home/golah/wolf_camtrap/scripts/classification/temp/4_wolf_camtrap_EfficientNetV2L_unfrozen_best_lr00001.keras" # Combined
#model = tf.keras.models.load_model(model_path)
#classes = sorted(os.listdir("/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/train/"))
#print(classes)
#prediction_path = "/home/golah/wolf_camtrap/background_test/NonE_train27_25/labels/"

# Order models
model_path_c = "/home/golah/wolf_camtrap/scripts/classification/temp/2_wolf_camtrap_carnivora_EfficientNetV2L_unfrozen_best_lr00001.keras" # Carnivora
model_path_a = "/home/golah/wolf_camtrap/scripts/classification/temp/2_wolf_camtrap_artiodactyl_EfficientNetV2L_unfrozen_best_lr00001.keras" # Artio
model_c = tf.keras.models.load_model(model_path_c)
model_a = tf.keras.models.load_model(model_path_a)
classes_c = sorted(os.listdir("/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/test/"))
print(classes_c)
classes_a = sorted(os.listdir("/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/test_artio/"))
print(classes_a)
prediction_path = "/home/golah/wolf_camtrap/background_test/NonE_train28_25/labels/"

image_path      = "/home/golah/wolf_camtrap/training_2023/test/images/"
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
                                cropped_image = cv2.resize(cropped_image, (380, 380))
                                cropped_image = img_to_array(cropped_image) 
                                cropped_image = np.expand_dims(cropped_image, axis = 0)
                                cropped_image = cropped_image / 255.
 
                                # Run EfficientNetV2L prediction
                                # Species model
                                #classification_results = model.predict(cropped_image)
                                #classification_results = classes[np.argmax(classification_results)]
                                #frame_data.append([str(classification_results)])

                                # Order model
                                if float(label) == 0:
                                    classification_results = model_a.predict(cropped_image)
                                    classification_results = classes_a[np.argmax(classification_results)]
                                elif float(label) == 3:
                                    classification_results = model_c.predict(cropped_image)
                                    classification_results = classes_c[np.argmax(classification_results)]
                                frame_data.append([str(classification_results)])
                            else:
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
                print(file_df)
                most_frequent_label = file_df['label'].value_counts()[:1].index.tolist()[0]
                print(file_df['label'].value_counts())
                print(most_frequent_label)
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

prediction_results  = analyze_sequence(prediction_path, result_path + "EfficientNet_seq_group_predicted_results.csv")
print(prediction_results)


