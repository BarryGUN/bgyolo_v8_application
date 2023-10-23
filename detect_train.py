from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8-ms2-c2repx-sppfcsp.yaml')
    model.train(data='vlpd.yaml',
                epochs=500,
                project='/runs/detect/train',
                name='yolov8-ms2-c2repx-v2-sppfcsp-bs16',
                batch=16,
                cos_lr=True)
