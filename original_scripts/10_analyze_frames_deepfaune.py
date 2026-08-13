# Load libraries
import numpy as np
from PytorchWildlife.models import classification as pw_classification
import cv2
import os 
import pandas as pd

classification_model = pw_classification.DeepfauneClassifier() # Model weights are automatically downloaded.
prediction_path = "/home/golah/wolf_camtrap/background_test/NonE_DeepFaune_25/labels/"
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
                    cropped_image = cv2.resize(cropped_image, (224, 224))
                    #cv2.imshow("ASD", cropped_image)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows() 
                    # Run deepfaune classification model
                    classification_results = classification_model.single_image_classification(cropped_image)
                    label = classification_results['prediction']
                    #print("Before: ", label)
                    # Use scientific names
                    if label == "badger":
                        label = "Meles_meles"
                    elif label == "red deer":
                        label = "Cervus_elaphus"
                    elif label == "cat":
                        label = "Felis_silvestris"
                    elif label == "roe deer":
                        label = "Capreolus_capreolus"
                    elif label == "dog":
                        label = "Dog"
                    elif label == "wolf":
                        label = "Canis_lupus"
                    elif label == "mouflon":
                        label = "Ovis_orientalis"
                    elif label == "fox":
                        label = "Vulpes_vulpes"
                    elif label == "wild boar":
                        label = "Sus_scrofa"
                    else:
                        label = "Animal"
                    #print("After: ", label)
                    frame_data.append([str(label)])
                else:
                    print("ASD")
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

prediction_results  = analyze_frames(prediction_path, result_path + "DeepFaune_frame_class_predicted_results.csv")
print(prediction_results)

