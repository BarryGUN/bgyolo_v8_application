from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('w/dense/best.pt')
    # model = YOLO('w/v8/best.pt')
    # results = model(r'C:\Users\lenovo\Desktop\car-drive\01.mp4', device='0', half=False, show=True, conf=0.4, iou=0.6)


    results = model('demo/004.jpg',
                    device='0',
                    half=False,
                    show=False,
                    save=True,
                    conf=0.22,
                    # conf=0.208,
                    iou=0.6,
                    project='runs/val/n/',
                    name='densefpn',
                    # name='v8',
                    plots=True)
