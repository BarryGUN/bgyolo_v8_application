import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-vis.pt') # weight
    model.val(project='../runs/detect/val/n',  # save prefix root "../ + Roadbed from the content root."
              name='yolov8n-vis',   # save folder name
              half=True,    # half datatype
              plots=True,
              iou=0.6)

