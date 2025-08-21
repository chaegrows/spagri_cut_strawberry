import sys
from ultralytics import YOLO

model = YOLO(sys.argv[1])
model.export(imgsz=(480, 640), format="engine")
