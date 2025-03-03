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

## How to use

**NOTE:** Tested on Windows 11 OS

### Install pip packages

```sh
pip.exe install -r requirements.txt
```

### Run object detector application

```sh
python.exe object_detector_app.py --help

# Object detection with pretrained nano YOLO V8 model
python.exe object_detector_app.py --model nano_yolov8

# Object detection with pretrained nano YOLO 11 model
python.exe object_detector_app.py --model nano_yolov11

# Object detection with pretrained large DETR model
python.exe object_detector_app.py --model large_detr
```
