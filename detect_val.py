from ultralytics import YOLO



if __name__ == '__main__':
    # model = YOLO('runs/detect/train/n/yolov8n-bdd100k-bs16-linear-adamw/weights/best.pt')
    model = YOLO('runs/detect/train/n/yolov8n-bdd100k-densefpn-eiou-bs16-linear-adamw/weights/best.pt')
    model.val(project='runs/detect/val/n', name='yolov8n-bdd100k-densefpn-eiou-bs16', half=True, plots=True, iou=0.6)

    # model = YOLO('yolov8n.pt')
    # model.val(project='runs/detect/val', name='yolov8n-coco', half=False, plots=True, data='coco.yaml')