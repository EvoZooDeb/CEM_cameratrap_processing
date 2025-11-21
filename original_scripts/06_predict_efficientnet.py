# Load libraries
import numpy as np
import cv2
import os 
import pandas as pd
import tensorflow as tf
from keras.utils import img_to_array
import random 
import datetime

model_path = "/home/golah/wolf_camtrap/scripts/classification/temp/4_wolf_camtrap_EfficientNetV2L_unfrozen_best_lr00001.keras" # Combined
model_path_c = "/home/golah/wolf_camtrap/scripts/classification/temp/2_wolf_camtrap_carnivora_EfficientNetV2L_unfrozen_best_lr00001.keras" # Carnivora
model_path_a = "/home/golah/wolf_camtrap/scripts/classification/temp/2_wolf_camtrap_artiodactyl_EfficientNetV2L_unfrozen_best_lr00001.keras" # Artio
model = tf.keras.models.load_model(model_path)
model_c = tf.keras.models.load_model(model_path_c)
model_a = tf.keras.models.load_model(model_path_a)

prediction_path = "/home/golah/wolf_camtrap/training_2023/yolo_annotations_test_4_combined/"
image_path      = "/home/golah/wolf_camtrap/training_2023/test/images/"
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

classes = sorted(os.listdir("/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/train/"))
print(classes)
classes_c = sorted(os.listdir("/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/test/"))
print(classes_c)
classes_a = sorted(os.listdir("/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/test_artio/"))
print(classes_a)

file_names   = os.listdir(prediction_path)
print(len(file_names))

# Random sample 300 animal images for runtime test
from itertools import compress
file_names = list(compress(file_names, ["Vehicle" not in i for i in file_names]))
file_names = list(compress(file_names, ["Person" not in i for i in file_names]))
print(len(file_names))
random.seed(123)
frame_names = random.sample(file_names, 300)
print(frame_names)

# For each file in directory
start_time = datetime.datetime.now()
for frame in frame_names:
    print("Working with frame: ", frame, "--------------------------------------------------------------------------------------")
    image_name = frame.split(".",)[0] + ".JPG"
    image = cv2.imread(image_path + image_name)
    h_image, w_image = image.shape[:2]
    with open(prediction_path + frame) as f:
        for line in f.readlines():
            label, x_center, y_center, w_box, h_box = line.rstrip().split(" ")
            # If animal is in the bounding boxi
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
                #cv2.imshow("ASD", cropped_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                cropped_image = img_to_array(cropped_image) 
                cropped_image = np.expand_dims(cropped_image, axis = 0)
                cropped_image = cropped_image / 255.
                
                # Run EfficientNetV2L prediction
                # Species only model
                #classification_results = model.predict(cropped_image)
                #classification_results = classes[np.argmax(classification_results)]

                # Based on order
                # Taking account previous group:
                if float(label) == 0:
                    classification_results = model_a.predict(cropped_image)
                    classification_results = classes_a[np.argmax(classification_results)]
                elif float(label) == 3:
                    classification_results = model_c.predict(cropped_image)
                    classification_results = classes_c[np.argmax(classification_results)]
end_time = datetime.datetime.now()
print("Start time:" , start_time, "End time: ", end_time)
print("Time elapsed: ", end_time - start_time)

