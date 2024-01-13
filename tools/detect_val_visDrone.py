import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-vis.pt')              # weight address
    model.val(project='../runs/detect/val/n',   # save prefix root "../ + Roadbed from the content root."
              name='yolov8n-vis',               # save folder name
              half=True,                        # half Precision
              plots=True,                       # plot result
              save_json=True,                   # save json for cocoeval
              split='test',                     # data set type test, train, val
              iou=0.6,                          # iou thresh
              device=0,                         # gpu use '0 or [0 ,1 ,2 ,3, i]' , cpu use 'cpu'
              data='VisDrone.yaml'              # data set yaml
              )

