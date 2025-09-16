# Load libraries
import cv2 
import numpy as np
import os 
import pandas as pd
import datetime as dt
from dateutil.parser import parse

dir_name  = "20180307_bzs4b"
camera_id = "bzs4b"
footage_date = "20180307"

dir_path   = "/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/results/" + dir_name + "/" + camera_id + "/" + footage_date + "/"
result_path   = "/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/results/" + dir_name + "/" + camera_id + "/results/"
#exif_path

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
  12: "Dama dama",
  13: "Lepus europaeus",
  14: "Martes martes",
  15: "Buteo buteo",
  16: "Lutra lutra",
}

file_data  = []
# For each file in directory
for dir_index, directory in enumerate(os.listdir(dir_path)):
    print("Working in dir: ", directory, dir_index, "--------------------------------------------------------------------------------------")
    frame_data = []
    group_data = []
    if dir_index < 0:
        continue
    else:
        annotation_path = dir_path + directory + "/labels/"
        for file in os.listdir(annotation_path):
            with open(annotation_path + file) as f:
                group_size = 0
                group_size_add = False
                for line in f.readlines():
                    label, x_center, y_center, width, height, conf = line.rstrip().split(" ")
                    if float(conf) > 0.25:
                        group_size = group_size + 1
                        frame_data.append([f"{names.get(int(label))}"])
                        group_size_add = True
                if group_size_add == True:
                    group_data.append(group_size)
        if(len(frame_data) > 0):
            frame_df = pd.DataFrame(frame_data, columns = ['label'])
            most_frequent_label = frame_df['label'].value_counts()[:1].index.tolist()[0]
            most_frequent_label_count = frame_df[frame_df['label'] == most_frequent_label].count().iloc[0]
            number_of_annotated_objects = frame_df.count().iloc[0]
            number_of_annotated_frames = len(group_data)
            confidence = most_frequent_label_count / number_of_annotated_objects
            mean_group_size = sum(group_data)/len(group_data)
            max_group_size  = max(group_data)
            file_data.append([camera_id, footage_date, directory, f"{names.get(int(label))}", confidence, str(number_of_annotated_objects) ,str(number_of_annotated_frames), mean_group_size, max_group_size])
        else:
            file_data.append([camera_id, footage_date, directory, 'EMPTY', 0, 0, 0, 0, 0])
file_df = pd.DataFrame(file_data, columns = ['camera_id', 'footage_date', 'file_name', 'label', 'label_confidence','n_annotated_objects' ,'n_annotated_frames', 'mean_group_size', 'max_group_size'])

# Only if video
#print(file_df)
#file_df.to_csv(result_path + dir_name + "_" + camera_id + "_" + footage_date + "_" + "results.csv", sep = ",", encoding = "utf-8")

# Only if images
# Post-process sequence data
def convert_to_datetime(string):
    return dt.datetime(int(string[:4]), int(string[5:7]), int(string[8:10]), int(string[11:13]), int(string[14:16]), int(string[17:19]))

sequence_data = []
exif_data = pd.read_csv(result_path + dir_name + "_" + camera_id + "_" + footage_date + "_exif.csv")
exif_data["file_name"] = [i.split(".",)[0] for i in exif_data['file_name']]
exif_data_sorted  = exif_data.sort_values('date_exif')
b = [convert_to_datetime(i) for i in exif_data_sorted['date_exif']]
k = 0
for i in range(0, len(b), 1):
    if i == k:
        l = 1
        anchor_time = b[i]
        for j in range(i+1, len(b), 1):
            compare_time = b[j]
            c = (compare_time - anchor_time).total_seconds()
            if c > 10:
                break
            else:
                l = l + 1
        
        k = 0 + j
        # Image
        if k + 1 < len(b) and l > 1:
            sequence_df = exif_data_sorted.iloc[i:j]
        elif l > 1:
            sequence_df = exif_data_sorted.iloc[i:j+1]
            k = k + 1
        elif k + 1 <= len(b) and l == 1:
            sequence_df = exif_data_sorted.iloc[i:i+1]
        else:
            sequence_df = exif_data_sorted.iloc[i:j]

       
       # print(sequence_df) 
       # Video
       # elif i+1 == len(b) and j == i:
       #     sequence_df = exif_data_sorted.iloc[i:i+1]
       # elif j == i+1:
       #     sequence_df = exif_data_sorted.iloc[i:j]

        joined_df = pd.merge(sequence_df, file_df, how = "left", on=["file_name"])
        joined_df = joined_df.sort_values('file_name')
        #print(joined_df)
        if(len(joined_df) == 1):
            joined_date = joined_df['date_exif'].iloc[0]
            duration = joined_df['duration'].iloc[0]
            joined_file_name = joined_df['file_name'].iloc[0]
            most_frequent_label = joined_df['label'][0]
            number_of_annotated_objects = joined_df['n_annotated_objects'][0]
            number_of_annotated_frames  = joined_df['n_annotated_frames'][0]
            confidence = joined_df['label_confidence'][0]
            mean_group_size = joined_df['mean_group_size'][0]
            max_group_size  = joined_df['max_group_size'][0]
            sequence_data.append([camera_id, footage_date, joined_date, duration, joined_file_name, most_frequent_label, confidence, str(number_of_annotated_objects) ,str(number_of_annotated_frames), mean_group_size, max_group_size, ""])
        elif(len(joined_df) > 1):
            joined_date = joined_df['date_exif'].iloc[0] + "-" + joined_df['date_exif'].iloc[len(joined_df)-1][17:19]
            duration = len(joined_df)
            joined_file_name = joined_df['file_name'].iloc[0] + "-" + joined_df['file_name'].iloc[len(joined_df)-1][4:]
            joined_df = joined_df[joined_df['label'] != "EMPTY"]
            if len(joined_df) > 0:
            # Többség alapján döntés vagy valami konfidenciával súlyozott érték alapján?
                most_frequent_label = joined_df['label'].value_counts()[:1].index.tolist()[0]
                most_frequent_label_count = joined_df[joined_df['label'] == most_frequent_label].count().iloc[0]
                number_of_annotated_objects = joined_df['n_annotated_objects'].astype(int).sum()
                number_of_annotated_frames = len(joined_df)
                confidence = most_frequent_label_count / number_of_annotated_frames
                mean_group_size = sum(joined_df['max_group_size'])/ number_of_annotated_frames
                max_group_size  = max(joined_df['max_group_size'])
                sequence_data.append([camera_id, footage_date, joined_date, duration, joined_file_name, most_frequent_label, confidence, str(number_of_annotated_objects) ,str(number_of_annotated_frames), mean_group_size, max_group_size, ""])
            else:
                sequence_data.append([camera_id, footage_date, joined_date, duration, joined_file_name, 'EMPTY', 0, 0, 0, 0, 0, ""])
        else:
            sequence_data.append([camera_id, footage_date, joined_date, joined_file_name, 'EMPTY', 0, 0, 0, 0, 0, ""])

    else:
        continue

sequence_df = pd.DataFrame(sequence_data, columns = ['camera_id', 'upload_date', 'sequence_date', 'sequence_duration','sequence_file_names', 'label', 'label_confidence','n_annotated_objects' ,'n_annotated_frames', 'mean_group_size', 'max_group_size', 'comment'])
print(sequence_df)
sequence_df.to_csv(result_path + dir_name + "_" + camera_id + "_" + footage_date + "_" + "results.csv", sep = ",", encoding = "utf-8")
