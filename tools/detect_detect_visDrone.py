import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-vis.pt')  # weight address
    results = model(source='',      # detect source, support video,cam, images
                    device='0',     # gpu use '0 or [0 ,1 ,2 ,3, i]' , cpu use 'cpu'
                    half=True,      # half Precision
                    show=True,      # show images or video
                    conf=0.32,      # conf thresh
                    iou=0.6)        # iou thresh



