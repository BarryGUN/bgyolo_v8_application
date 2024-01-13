from ultralytics.utils.benchmarks import benchmark

if __name__ == "__main__":
    benchmark(model='yolov8n-vis.pt',   # weight address
              data='VisDrone.yaml',     # data set yaml
              imgsz=640,                # image size
              half=True,                # half Precision
              device=0                  # gpu use '0 or [0 ,1 ,2 ,3, i]' , cpu use 'cpu'
              )