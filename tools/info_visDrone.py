import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('visDrone/yolov8n-vis.yaml')   # cfg in Roadbed from the content root: ultralytics/cfg/models/v8/visDrone
    """
        n ---- visDrone/yolov8n-vis.yaml
        s ---- visDrone/yolov8s-vis.yaml
        m ---- visDrone/yolov8m-vis.yaml
    """
    model.info(
        detailed=False      # if you want see layer detail, set 'True'
    )