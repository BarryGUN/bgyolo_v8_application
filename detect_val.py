from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('runs/detect/train/m/yolov8m-ms2d-c2repx-vlpd-bs16/weights/best.pt')
    model.val(project='runs/detect/val/m', name='yolov8m-ms2d-c2repx-vlpd-bs16', half=True, plots=True)

    # model = YOLO('yolov8n.pt')
    # model.val(project='runs/detect/val', name='yolov8n-coco', half=False, plots=True, data='coco.yaml')