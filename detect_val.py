from ultralytics import YOLO



if __name__ == '__main__':
    # model = YOLO('runs/detect/train/yolov8n-vlpd-bs16/weights/best.pt')
    # model.val(project='runs/detect/val', name='yolov8n-vlpd-bs16-fp16', half=False, plots=True)

    model = YOLO('yolov8n.pt')
    model.val(project='runs/detect/val', name='yolov8n-coco', half=False, plots=True, data='coco.yaml')