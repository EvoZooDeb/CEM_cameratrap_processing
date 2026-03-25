# Load libraries
import numpy as np
import cv2
import os 
import pandas as pd
import tensorflow as tf
from keras.utils import img_to_array

# Define paths
# Species model
model_path = "/path/to/species/model/4_wolf_camtrap_EfficientNetV2L_unfrozen_best_lr00001.keras" # Combined
model = tf.keras.models.load_model(model_path)

# Group models
#model_path_c = "/path/to/carnivora/model/2_wolf_camtrap_carnivora_EfficientNetV2L_unfrozen_best_lr00001.keras" # Carnivora
#model_path_a = "/path/to/artio/model/2_wolf_camtrap_artiodactyl_EfficientNetV2L_unfrozen_best_lr00001.keras" # Artio
#model_c = tf.keras.models.load_model(model_path_c)
#model_a = tf.keras.models.load_model(model_path_a)

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

# Animal classes 
classes = sorted(os.listdir("/training_classification/train/")) 

# General class names and codes
gen_names = {
  0: "Animal",
  1: "Person",
  2: "Vehicle",
}

# Group classes
#classes_c = sorted(os.listdir("/training_carnivora/train/"))
#print(classes_c)
#classes_a = sorted(os.listdir("/training_artio/train/"))
#print(classes_a)

def analyze_frames(path, result_path):
    file_names   = os.listdir(path)
    frame_names  = [file for file in file_names if "frame" in file] # Keep only videos
    results      = []
    
    # For each file in directory
    for frame in frame_names:
        print("Working with frame: ", frame, "--------------------------------------------------------------------------------------")
        image_name = frame.split(".",)[0] + ".JPG"
        image = cv2.imread(image_path + image_name)
        h_image, w_image = image.shape[:2]
        frame_data = []
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
                    # Species only model
                    classification_results = model.predict(cropped_image)

                    # TOP1
                    #classification_results = classes[np.argmax(classification_results)]

                    # TOP3
                    classification_results = np.argpartition(classification_results[0], -3)[-3:]
                    classification_results = classes[classification_results[0]] + "/" + classes[classification_results[1]] + "/" + classes[classification_results[2]]
                    
                    frame_data.append([str(classification_results)])
                    
                    # GRP - Based on order
                    # Taking account previous group:
                    #if float(label) == 0:
                    #    classification_results = model_a.predict(cropped_image)
                    #    classification_results = classes_a[np.argmax(classification_results)]
                    #elif float(label) == 3:
                    #    classification_results = model_c.predict(cropped_image)
                    #    classification_results = classes_c[np.argmax(classification_results)]
                    #frame_data.append([str(classification_results)])

                    # Run both models on general detector results and pick higher score - slower and less accurate
                    #classification_results_a = model_a.predict(cropped_image)
                    #score_a = np.max(classification_results_a)
                    #classification_results_c = model_c.predict(cropped_image)
                    #score_c = np.max(classification_results_c)
                    #if score_a > score_c:
                    #    classification_results = classes_a[np.argmax(classification_results_a)]
                    #elif score_c > score_a:
                    #    classification_results = classes_c[np.argmax(classification_results_c)]
                    #elif float(label) == 0:
                    #    classification_results = classes_a[np.argmax(classification_results_a)]
                    #else:
                    #    classification_results = classes_c[np.argmax(classification_results_c)]
                    #print(classification_results)
                    #frame_data.append([str(classification_results)])
                else:
                    frame_data.append([f"{gen_names.get(int(label))}"])
                group_size = group_size + 1
        frame_df = pd.DataFrame(frame_data, columns = ['label'])
        most_frequent_label = frame_df['label'].value_counts()[:1].index.tolist()[0]
        most_frequent_label_count = frame_df[frame_df['label'] == most_frequent_label].count().iloc[0]
        confidence = most_frequent_label_count / group_size
        results.append([frame, most_frequent_label, confidence, group_size])
    result_df = pd.DataFrame(results, columns = ['file_name', 'label', 'label_confidence', 'group_size'])
    result_df.to_csv(result_path, sep = ",", encoding = "utf-8")
    return result_df

prediction_results  = analyze_frames(prediction_path, result_path + "EfficientNet_frame_TOP3_predicted_results.csv")
print(prediction_results)

