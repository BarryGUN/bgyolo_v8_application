from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('yolov8m-ms2d-c2repx.yaml')
    model = YOLO('yolov8m-ms2e-c2repx.yaml')
    model.train(data='vlpd.yaml',
                epochs=500,
                project='runs/detect/train/m',
                name='yolov8s-vlpd-ms2d-c2repx-bs16',
                batch=16,
                cos_lr=True,
                amp=True
                )




