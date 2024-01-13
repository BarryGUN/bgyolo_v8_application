## <div align="center">YOLOv8-vis(YOLOVIS) T_OO_T</div>
<br>


YOLOVIS is an unmanned aerial vehicle (UAV) object detection model based on the YOLOv8 backbone and VIS head, trained on the VisDrone dataset.


## <div align="center">Quick Start</div>


<details open>
<summary>Conda install</summary>

create conda environment and enter
```bash
conda create -n yolovis python=3.8
conda activate yolovis
```
change your path to code 
```bash
cd <path to your code> 
```
install packages
```bash
pip install -r requirements.txt
```


</details>

<details open>
<summary>Base install</summary>

change your path to code 
```bash
cd <path to your code> 
```
install packages
```bash
pip install -r requirements.txt
```


</details>

<details open>
<summary>train</summary>

Train our model 
```bash
python tools/detect_train_visDrone.py
```
For detailed parameters, refer to tools/detect_train_visDrone.py

</details>

<details open>
<summary>test</summary>

Test(val) our model 
```bash
python tools/detect_val_visDrone.py
```
For detailed parameters, refer to tools/detect_val_visDrone.py

</details>

<details open>
<summary>detect</summary>

inference our model 
```bash
python tools/detect_detect_visDrone.py
```
For detailed parameters, refer to tools/detect_detect_visDrone.py

</details>

<details open>
<summary>benchmark</summary>

benchmark our model 
```bash
python tools/detect_benchmark_visDrone.py
```
For detailed parameters, refer to tools/detect_benchmark_visDrone.py

</details>

<details open>
<summary>model info</summary>

show info of  our model 
```bash
python tools/info_visDrone.py
```
For detailed parameters, refer to tools/nfo_visDrone.py

</details>

## <div align="center">Model</div>

<details open><summary>detect</summary>



| Model        | size | mAP<sup>test<br>50 | speed<br><sup>RTX3060 TensorRT<br>(ms) | param<br><sup>(M) | FLOPs<br><sup>(G) |
|--------------|------|--------------------|----------------------------------------|-------------------|-------------------|
| [YOLOVISn]() | 640  | 26.3               | 4.74                                   | 3.5               | 9.7               |
| [YOLOVISs]() | 640  | 30.1               | 5.38                                   | 13.0              | 34.6              |
| [YOLOVISm]() | 640  | 33.7               | 6.58                                   | 29.2              | 90.7              |



- **mAP<sup>test</sup>** This is based on the results of a single-model scale on the [VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) dataset.
  <br>Reproduce using `python tools/detect_val_visDrone.py`
- **speed** The speed is tested on the VisDrone test using benchmark.py, with a batch size of 1.
  <br>Reproduce using `python tools/benchmark.py`

</details>

## <div align="center">More Detail</div>
- Our model is based on YOLOv8's constructor, see [ultralytics](https://github.com/ultralytics/ultralytics) for more details

