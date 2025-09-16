from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
import datetime

dir_name  = "20180307_bzs4b"
camera_id = "bzs4b"
footage_date = "20180307"

image_path   = "/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/raw/" + dir_name + "/Camera_footage/" + camera_id + "/" + footage_date + "/"
project_path = "/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/results/" + dir_name + "/"
project_name = camera_id + "/" + footage_date

model = YOLO("/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/models/best_26.pt")

# Multiple file type
print("DETECTING.")
print("DETECTING..")
print("DETECTING...")
start_time = datetime.datetime.now()
for file in os.listdir(image_path):
    dynamic_project_name = project_name + "/" + file.split(".",)[0]
    image_full_path = image_path + file
    if file.endswith(('jpg', 'JPG', 'png', 'PNG')):
        results = model.predict(image_full_path, save = False, save_txt = True, save_conf = True, imgsz = (1280), conf = 0.25, iou = 0.45, agnostic_nms = True, project =project_path, name = dynamic_project_name, device = "cpu") # predict on an image
    elif file.endswith(('mp4', 'MP4', 'avi', 'AVI', 'MOV', 'mov')):
        results = model.predict(image_full_path, save = False, save_frames = False, save_txt = True, save_conf = True, show_labels = False, show_boxes = False, show_conf = False, imgsz = (1280), conf = 0.25, iou = 0.45, augment = True, agnostic_nms = True, stream = True, project = project_path, name = dynamic_project_name, device = "cuda:0") # predict on video
        for r in results:
            pass
    print(file + " DONE!")
end_time = datetime.datetime.now()
print("Start time:" , start_time, "End time: ", end_time)
print("Time elapsed: ", end_time - start_time)
print("DETECTING ENDED")

# POST PROCESS
print("POST PROCESSING.")
print("POST PROCESSING..")
print("POST PROCESSING...")
#project_dir = project_path + project_name + "/"
#for directory in os.listdir(project_dir):
#    image_path = project_dir + directory + "/" + directory + "_frames/"
#    annotation_path = project_dir + directory + "/" + "labels/"
#    os.remove(project_dir + directory + "/" + directory + ".avi")
#    for frame in os.listdir(image_path):
#        annotation_name = directory + "_" + frame.split(".",)[0] + ".txt"
#        if not os.path.exists(annotation_path + annotation_name):
#            os.remove(image_path + frame)
print("POST PROCESSING ENDED")



