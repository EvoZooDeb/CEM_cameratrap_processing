from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
import datetime

# Define paths
image_path   = "/image/path/"
project_path = "/result/path/"
project_name = "NonE_DeepFaune_25"
model = YOLO("/model/path/best.pt")
#model = YOLO('/model/path/deepfaune-yolov8s_960.pt')
#model = YOLO('/model/path/MDV6-yolov10x.pt')

start_time = datetime.datetime.now()
for result in model.predict(image_path, stream = True, save = False, save_txt = False, imgsz = (1280), conf = 0.25, iou = 0.45, agnostic_nms = True, project = project_path, name = project_name, device = "cuda:0"): # predict on an image
    pass
end_time = datetime.datetime.now()
print("Start time:" , start_time, "End time: ", end_time)
print("Time elapsed: ", end_time - start_time)
