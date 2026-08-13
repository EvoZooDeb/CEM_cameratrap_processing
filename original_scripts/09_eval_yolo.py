from ultralytics import YOLO

# Load a model
model = YOLO('/home/golah/wolf_camtrap/scripts/runs/detect/train28/weights/best.pt')  # Detect
#model = YOLO('/home/golah/wolf_camtrap/base_neural_models/deepfaune/deepfaune-yolov8s_960.pt')  # Detect
#model = YOLO('/home/golah/wolf_camtrap/base_neural_models/megadetector_v6-10e/MDV6-yolov10x.pt')  # Detect

# Validate detection the model
#metrics = model.val(data = "/home/golah/wolf_camtrap/training_2023/camtrap_yolov8.yaml", conf = 0.25, iou = 0.45, device = 'cuda:0', split ='test', save_json = True, imgsz = 1280) 
metrics = model.val(iou = 0.45, device = 'cuda:0', split ='val', save_json = True)
F1_score = 2*(metrics.box.mp*metrics.box.mr)/(metrics.box.mp+metrics.box.mr)
print("MAP50", metrics.box.map50)
print("MAP50-95", metrics.box.map)
print("PRECISION", metrics.box.mp)
print("RECALL", metrics.box.mr)
print("F1_SCORE", F1_score)

