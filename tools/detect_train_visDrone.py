import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('visDrone/yolov8s-vis.yaml')       # cfg in Roadbed from the content root: ultralytics/cfg/models/v8/visDrone
    """
           n ---- visDrone/yolov8n-vis.yaml
           s ---- visDrone/yolov8s-vis.yaml
           m ---- visDrone/yolov8m-vis.yaml
           
    """
    model.train(data='VisDrone.yaml',               # data set yaml
                epochs=600,                         # epoch
                project='../runs/detect/train/n',   # save prefix root "../ + Roadbed from the content root."
                name='yolov8-vis',  # save folder name
                batch=16,            # batch size
                cos_lr=False,       # use Cosine lr, if you want open it, set True
                amp=True,           # use amp
                optimizer='AdamW',  # optimizer, support[SGD, Adam, AdamW]
                lr0=0.001,          # start learning rate AdamW 0.001, SGD 0.01
                lrf=0.01,           # final learning rate = lr0*lrf
                iou=0.60,           # iou thresh
                conf=0.001,         # conf
                pretrained=False,   # no pretrain model
                resume=False,
                seed=0,
                deterministic=True,     # "Ensure Reproducibility." need pytorch 2.0+
                patience=100,           # early stop epochs
                device=0,               # gpu use '0 or [0 ,1 ,2 ,3, i]' , cpu use 'cpu'
                imgsz=640               # image size
                )




