from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    model.val(data='coco128.yaml', epochs=180, half=False)