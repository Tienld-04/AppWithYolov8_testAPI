import numpy as np
from ultralytics import YOLO

print(np.__version__)
print("Le Danh Tien")
model = YOLO("yolov8n.pt")
