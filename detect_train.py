from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('yolov8m-ms2d-c2repx.yaml')
    model = YOLO('bdd100k/yolov8n-bincat.yaml')
    # model = YOLO('yolov8n.yaml')
    model.train(data='coco128.yaml',
                epochs=500,
                project='runs/detect/train/m',
                name='yolov8s-vlpd-ms2e-c2repx-bs16-v2',
                batch=2,
                cos_lr=True,
                amp=True
                )




