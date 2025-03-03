# python-object-detection-with-ultralytics-ai-models-and-opencv

## Table of Contents

+ [Summary](#summary)
+ [References](#references)
+ [How to use](#how-to-use)

## Summary

Simple Object detector app utilizing [ultralytics AI models](https://docs.ultralytics.com/models/). The detector primarily utilizes RTDETR, YOLOv8 and YOLO11 models.

## References

- [YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/#usage)
- [Model Prediction with Ultralytics YOLO](https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode)
- [Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector](https://docs.ultralytics.com/models/rtdetr/)
- [rtdetrClass.py from niconielsen32](https://github.com/niconielsen32/DETR/blob/main/rtdetrClass.py)

## Preconditions

- Setup Python trough an OS specific package manager (`brew` on Mac OS and `choco` on Windows) or download release from [from the official website]()

## How to use

**NOTE:** Tested on Windows 11 OS and Mac OS 15

### Install pip packages

Run following commands:

```sh
# Windows 11 OS
pip.exe install -r requirements.windows11.txt

# Mac OS 15 (NOTE: consider --break-system-packages only in case you encounter 'error: externally-managed-environment')
pip3 install -r requirements.macos15.txt --break-system-packages
```

### Run object detector application

Run following commands on Windows 11 OS:

```sh
python.exe object_detector_app.py --help

# Object detection with pretrained nano YOLO V8 model
python.exe object_detector_app.py --model nano_yolov8

# Object detection with pretrained nano YOLO 11 model
python.exe object_detector_app.py --model nano_yolov11

# Object detection with pretrained large DETR model
python.exe object_detector_app.py --model large_detr
```

Run following commands on Mac OS 15:

```sh
python3 object_detector_app.py --help

# Object detection with pretrained nano YOLO V8 model
python3 object_detector_app.py --model nano_yolov8

# Object detection with pretrained nano YOLO 11 model
python3 object_detector_app.py --model nano_yolov11

# Object detection with pretrained large DETR model
python3 object_detector_app.py --model large_detr
```