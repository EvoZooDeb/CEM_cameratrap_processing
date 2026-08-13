# Load libraries
import json
import os
import cv2

# Define static paths

# Classifier
annotation_path = "/home/golah/wolf_camtrap/training_2023/yolo_annotations_train/"
image_path      = "/home/golah/wolf_camtrap/training_2023/train/images/"
crop_path       = "/home/golah/wolf_camtrap/training_classification_2024/standard_box_size/train/"

# For each training image
for index, file in enumerate(os.listdir(image_path)):
    if index < 0:
        continue
    else:
        print("Working with file: ", file, index)
        annotation_name      = file.split(".",)[0] + ".txt"
        annotation_full_path = annotation_path + annotation_name
        image_name           = file.split(".",)[0]
        image                = cv2.imread(image_path + file)
        h_image, w_image     = image.shape[:2]
        
        # If the corresponding annotation is found
        if os.path.exists(annotation_full_path):
            
            # Load the .txt annotation with the same name
            with open(annotation_full_path) as f:
            
            # For each animal/object detected crop bbox by YOLO coordinates
                for i, line in enumerate(f.readlines()):
                    label, x_center, y_center, w_box, h_box = line.rstrip().split(" ")
                    label = float(label) 
                    # Get the species name based on label encoding
                    if label == 0:
                        species = "Canis_lupus"
                    elif label == 1:
                        #species = "Person"
                        continue
                    elif label == 2:
                        species = "Cervus_elaphus"
                    elif label == 3:
                        species = "Capreolus_capreolus"
                    elif label == 4:
                        species = "Sus_scrofa"
                    elif label == 5:
                        species = "Vulpes_vulpes"
                    elif label == 6:
                        species = "Felis_silvestris"
                    elif label == 7:
                        species = "Meles_meles"
                    elif label == 8:
                        species = "Ovis_orientalis"
                    elif label == 9:
                        #species = "Vehicle"
                        continue
                    elif label == 10:
                        species = "Dog"

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
                    if h_cropped > 380 or h_cropped < 380:
                        cropped_image = cv2.resize(cropped_image, (380, 380))
                    
                    # Include more background on small images, instead of resizing - but sometimes it's almost the whole image
                    #elif h_cropped < 380:
                    #    crop_size = 190
                    #    x_min = int(x_center_px - crop_size)
                    #    x_max = int(x_center_px + crop_size)
                    #    y_min = int(y_center_px - crop_size)
                    #    y_max = int(y_center_px + crop_size)
                    #    print("ASD", x_min, x_max, y_min, y_max)
                    #    if x_min < 0:
                    #        print("first")
                    #        x_max = x_max - x_min
                    #        x_min = 0
                    #    if x_max > w_image:
                    #        print("second")
                    #        x_min = x_min - (x_max - w_image)
                    #        x_max = w_image
                    #    if y_min < 0:
                    #        print("third")
                    #        y_max = y_max - y_min
                    #        y_min = 0
                    #    if y_max > h_image:
                    #        print("fourth")
                    #        y_min = y_min - (y_max - h_image)
                    #        y_max = h_image
                    #    print(x_min, x_max, y_min, y_max)
                    #    cropped_image = image[y_min:y_max, x_min:x_max]

                    h_cropped, w_cropped = cropped_image.shape[:2]
                    #cv2.imshow("ASD", cropped_image)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()

                    # Save cropped image
                    cv2.imwrite(crop_path + species + "/" + image_name + "_object_" + str(i) + ".JPG", cropped_image)

        print("Image DONE")
    


