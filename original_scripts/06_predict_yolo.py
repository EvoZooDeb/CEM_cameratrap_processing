from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
import datetime

#dir_name = "Kleszo_Also_szoros_202201"
#dir_name = "Kleszo_Farkas_ko_202103"
#dir_name = "Pentek_Almas_Ret_20210206_0224"

image_path  = "/home/golah/wolf_camtrap/testing_2024/unknown_sites/raw_footage/Pentek_Almas_Ret_20210206_0224/"
#image_path  = "/home/golah/wolf_camtrap/training_2023/background_images_validated_KLESZO/"
#image_path  = "/home/golah/wolf_camtrap/training_2023/test/images/"
#image_path  = "/home/golah/wolf_camtrap/testing_2024/unknown_sites/sample_frames/" + dir_name + "/"
#image_path  = "/home/golah/wolf_camtrap/testing_2024/unknown_sites/raw_footage/" + dir_name + "/"

#project_path  = "/home/golah/wolf_camtrap/testing_2024/unknown_sites/yolo_labels/" + dir_name + "/"
project_path  = "/home/golah/wolf_camtrap/background_test/"
project_name = "NonE_DeepFaune_25"

model = YOLO("/home/golah/wolf_camtrap/scripts/runs/detect/train26/weights/best.pt")
#model = YOLO('/home/golah/wolf_camtrap/base_neural_models/deepfaune/deepfaune-yolov8s_960.pt')  # Detect
#model = YOLO('/home/golah/wolf_camtrap/base_neural_models/megadetector_v6-10e/MDV6-yolov10x.pt')  # Detect

start_time = datetime.datetime.now()
#results = model.predict(image_path, save = False, save_txt = True, imgsz = (1280), conf = 0.25, iou = 0.45, agnostic_nms = True, project = project_path, name = project_name, device = "cuda:0") # predict on an image
for result in model.predict(image_path, stream = True, save = False, save_txt = False, imgsz = (1280), conf = 0.25, iou = 0.45, agnostic_nms = True, project = project_path, name = project_name, device = "cuda:0"): # predict on an image
    pass
end_time = datetime.datetime.now()
print("Start time:" , start_time, "End time: ", end_time)
print("Time elapsed: ", end_time - start_time)

# Multiple file type
#for file in os.listdir(image_path):
#    image_full_path = image_path + file
#    
#    if file.endswith(('jpg', 'JPG', 'png', 'PNG')):
#        results = model.predict(image_full_path, save = False, save_txt = True, imgsz = (640), conf = 0.25, iou = 0.45, augment = True, agnostic_nms = True, project =project_path) # predict on an image
#    elif file.endswith(('mp4', 'MP4', 'avi', 'AVI', 'MOV', 'mov')):
#        results = model.predict(image_full_path, save = False, save_frames = False, save_txt = True, imgsz = (640,640), conf = 0.25, iou = 0.45, augment = True, agnostic_nms = True, stream = True, project = project_path, name = project_name) # predict on video
#        for r in results:
#            pass
#    print(file + " DONE!")
