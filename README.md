# python-object-detection-with-detr-and-opencv

## Table of Contents

+ [Summary](#summary)
+ [References](#references)
+ [How to use](#how-to-use)

## Summary

Simple Object detector app utilizing trained DETR models. The code in [object_detector_app.py](./object_detector_app.py) is based on [rtdetrClass.py from niconielsen32](https://github.com/niconielsen32/DETR/blob/main/rtdetrClass.py).

## References

- [rtdetrClass.py from niconielsen32](https://github.com/niconielsen32/DETR/blob/main/rtdetrClass.py)
- [Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector](https://docs.ultralytics.com/models/rtdetr/)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/#usage)

## How to use

### Install pip packages

```sh
pip.exe install -r requirements.txt
```

### Run object detector application

```sh
python.exe object_detector_app.py --help

# Object detection with pretrained nano YOLO V8 model
python.exe object_detector_app.py --model nano_yolov8

# Object detection with pretrained large DETR model
python.exe object_detector_app.py --model nano_yolov8
```