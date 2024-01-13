import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-vis.pt') # weight address
    results = model(source='',  # detect source video,cam, images
                    device='0',
                    half=True,
                    show=True,  # show images or video
                    conf=0.32,
                    iou=0.6)



