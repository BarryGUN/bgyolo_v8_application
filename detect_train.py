from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.yaml')
    model.train(data='coco2017-mini.yaml', epochs=180)