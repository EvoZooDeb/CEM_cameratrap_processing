# Load libraries
import cv2 
import numpy as np
import os 
from ultralytics.utils.plotting import Annotator, colors

dir_name  = "20180307_bzs4b"

camera_id = "bzs4b"

footage_date = "20180307"

dir_path   = "/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/results/" + dir_name + "/" + camera_id + "/" + footage_date + "/"

names = {
  0: "Canis lupus",
  1: "Person",
  2: "Cervus elaphus",
  3: "Capreolus capreolus",
  4: "Sus scrofa",
  5: "Vulpes vulpes",
  6: "Felis silvestris",
  7: "Meles meles",
  8: "Ovis orientalis",
  9: "Vehicle",
  10: "Dog",
  11: "Animal",
}
# Create a function which draws annotation on the given image/frame
def draw_annotation(image_path, image_name, index, annotation_name):
    print("Working with frame: ", image_name, index)
    if os.path.exists(annotation_path + annotation_name):
        fx = 1
        fy = 1
        image = cv2.imread(image_path + image_name)
        h, w = image.shape[:2]
        if 3000 >= h > 1500 and 3000 >= w > 1500:
            fx = 0.65
            fy = 0.65
            txs = 5
        elif 3000 >= h > 1500 or 3000 >= w > 1500:
            fx = 0.65
            fy = 0.65
            txs = 5
        elif h > 3000 or w > 3000:
            fx = 0.18
            fy = 0.18
            #fx = 0.25
            #fy = 0.25
            txs = 7.5
        ann = Annotator(
        image,
        line_width=None,  # default auto-size
        font_size=None,   # default auto-size
        font="Arial.ttf", # must be ImageFont compatible
        pil=False,        # use PIL, otherwise uses OpenCV
        )
        
        xyxy_boxes = []
        line_count = 0
        with open(annotation_path + annotation_name) as f:
            for line in f.readlines():
                label, x_center, y_center, width, height, conf = line.rstrip().split(" ")
                x_min = int(w * max(float(x_center) - float(width) / 2, 0))
                x_max = int(w * min(float(x_center) + float(width) / 2, 1))
                y_min = int(h * max(float(y_center) - float(height) / 2, 0))
                y_max = int(h * min(float(y_center) + float(height) / 2, 1))
                line_count = line_count + 1
                if float(conf) > 0.25:
                    xyxy_boxes.append([label, x_min, y_min, x_max, y_max, conf])
        if len(xyxy_boxes) != 0:
            for box in xyxy_boxes:
                c_idx, *box, c_conf = box
                label = f"{names.get(int(c_idx))}"
                ann.box_label(box, label+" "+c_conf, color=colors(c_idx, bgr=True))
             
            image_with_bboxes = ann.result()   
            if line_count > 1:
                image_with_bboxes = cv2.putText(image_with_bboxes, str(line_count), (round(w/2), 200), cv2.FONT_HERSHEY_SIMPLEX, txs, (0,0,255), 20)
            image_with_bboxes = cv2.resize(image_with_bboxes, (0,0), fx = fx, fy = fy)
            cv2.imshow((str(index) + "_" + image_name), image_with_bboxes) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Annotation not found!")

# For each file in directory
for dir_index, directory in enumerate(sorted(os.listdir(dir_path))):
    print("Working in dir: ", directory, dir_index, "--------------------------------------------------------------------------------------")
    if dir_index < 0:
        continue
    else:
        if os.path.exists(dir_path + directory + "/" + (directory + "_frames") + "/"):
            image_path       = dir_path + directory + "/" + (directory + "_frames") + "/"
            processing_video = True
        else:
            image_path = "/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/raw/" + dir_name + "/Camera_footage/" + camera_id + "/" + footage_date + "/"
            processing_video = False
        annotation_path = dir_path + directory + "/labels/"
        if processing_video == True:
            for index, file in enumerate(sorted(os.listdir(image_path))):
                annotation_name = directory + "_" + file.split(".",)[0] + ".txt"
                if index < 0:
                    continue
                else:
                    draw_annotation(image_path, file, index, annotation_name)
        else:
            annotation_name = directory.split(".",)[0] + ".txt"
            draw_annotation(image_path,  directory + ".JPG", dir_index, annotation_name)
