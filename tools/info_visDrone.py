import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('visDrone/yolov8n-vis.yaml') # cfg in Roadbed from the content root: ultralytics/cfg/models/v8/visDrone
    model.info()