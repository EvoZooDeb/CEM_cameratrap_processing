# Load libraries
import numpy as np
import cv2
import os 
import pandas as pd
import random 
import datetime
from PytorchWildlife.models import classification as pw_classification

model = pw_classification.DeepfauneClassifier()

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
                cropped_image = cv2.resize(cropped_image, (224, 224))
                
                # Species prediction
                classification_results = model.single_image_classification(cropped_image)

end_time = datetime.datetime.now()
print("Start time:" , start_time, "End time: ", end_time)
print("Time elapsed: ", end_time - start_time)

