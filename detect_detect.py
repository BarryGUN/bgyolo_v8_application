from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-bdd100k.pt')
    # results = model(r'C:\Users\lenovo\Desktop\car-drive\01.mp4', device='0', half=False, show=True, conf=0.4, iou=0.6)


    results = model('demo/001.jpg',
                    device='0',
                    half=False,
                    show=True,
                    conf=0.4,
                    iou=0.6,
                    visualize=True)
