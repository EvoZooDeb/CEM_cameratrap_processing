import os
import pandas as pd
import exifread
import ffmpeg
from datetime import datetime
import numpy as np


# Define static path
dir_name  = "20180307_bzs4b"
camera_id = "bzs4b"
footage_date = "20180307"
export_path = "/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/results/" + dir_name + "/" + camera_id +"/" + "results/"+ dir_name + "_" + camera_id + "_" + footage_date + "_exif.csv"
input_path  = "/home/gabor/Documents/motolla/hoi/kameracsapda/yolo/raw/" + dir_name + "/Camera_footage/" + camera_id + "/" + footage_date + "/"

exif_data = []
AVI_count = 0

# For each unique directory get exif info of all files inside directory
for file in os.listdir(input_path):
    input_full_path = os.path.join(input_path, file)
    print(input_full_path)
    if file.endswith(('jpg', 'JPG', 'png', 'PNG')):
        f = open(input_full_path, "rb")
        tags = exifread.process_file(f)
        date_time = tags['EXIF DateTimeOriginal'].printable
        duration = 1
    elif file.endswith(('MP4', 'mp4')):
        vid = ffmpeg.probe(input_full_path)
        date_time = vid['format']['tags']['creation_time']
        date_time = datetime.fromisoformat(date_time[:-1])
        duration  = vid['format']['duration']
    elif file.endswith(('AVI', 'avi')):
        #vid = ffmpeg.probe(input_full_path)
        AVI_count = AVI_count + 1
        continue
    exif_data.append([file, date_time, duration])
    print(file + " DONE!")

exif_df = pd.DataFrame(exif_data, columns = ['file_name', 'date_exif', 'duration'])
exif_df.to_csv(export_path, sep = ',', encoding = 'utf-8')
print("Exporting done: results saved to: ", export_path)
print("AVI count: ", AVI_count)





