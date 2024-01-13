from ultralytics.utils.benchmarks import benchmark

if __name__ == "__main__":
    benchmark(model='runs/detect/train/visDrone/n/yolov8n-visdrone-bs16-cos-adamw/weights/best.pt',
              data='VisDrone.yaml',
              imgsz=640,
              half=True,
              device=0)