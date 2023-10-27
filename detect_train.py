from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s-ms2d-c2repx.yaml')
    model.train(data='vlpd.yaml',
                epochs=500,
                project='runs/detect/train/s',
                name='yolov8s-vlpd-ms2d-c2repx-bs16',
                batch=16,
                cos_lr=True,
                amp=True
                )

