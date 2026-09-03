import cv2
from ultralytics import YOLO;

model = YOLO("yolov8n.pt")
results = model(source="video_test.mp4", show=True)  # show=True表示显示检测结果

print("END")