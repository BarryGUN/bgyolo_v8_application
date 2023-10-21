from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    results = model('C:\\Users\\lenovo\\Desktop\\object_detection\\test_data\\video_5.mp4', device='0',half=False)
