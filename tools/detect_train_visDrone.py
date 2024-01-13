import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('visDrone/yolov8s-vis.yaml')  # cfg in Roadbed from the content root: ultralytics/cfg/models/v8/visDrone
    model.train(data='VisDrone.yaml',
                epochs=600,
                project='../runs/detect/train/n',  # save prefix root "../ + Roadbed from the content root."
                name='yolov8-vis',  # save folder name
                batch=2,
                cos_lr=False,  # use Cosine lr, if you want open it, set True
                amp=True,   # use amp
                optimizer='AdamW',
                lr0=0.001,
                lrf=0.01,
                iou=0.65,
                conf=0.001,
                pretrained=False,
                resume=False,
                seed=0,
                deterministic=True,     # "Ensure Reproducibility." need pytorch 2.0+
                patience=100


                )




