from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('yolov8m-ms2d-c2repx.yaml')
    # model = YOLO('bdd100k/yolov8n-densefpn.yaml')
    # model = YOLO('runs/detect/train/n/yolov8n-bdd100k-densefpn-eiou-bs16-linear-adamw/weights/last.pt')

    # model.load('pretrain/train/n/yolov8n-vlpd-bs16/weights/best.pt')
    model = YOLO('bdd100k/yolov8n-densefpn-mf.yaml')
    model.train(data='coco128.yaml',
                epochs=600,
                project='runs/detect/train/n',
                name='yolov8n-bdd100k-densefpn-mf-eiou-bs16-linear-adamw',
                # name='yolov8n-bdd100k-densefpn-bs16-linear-adamw',
                batch=2,
                cos_lr=False,
                amp=True,
                optimizer='AdamW',
                lr0=0.001,
                lrf=0.01,
                iou=0.65,
                conf=0.001,
                pretrained=False,
                resume=True,
                seed=0,
                deterministic=True,
                wiou=True,
                eiou=False,
                alpha_regx=True,
                patience=100


                )




