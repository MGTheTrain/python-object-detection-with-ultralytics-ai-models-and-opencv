# python-object-detection-with-detr-and-opencv

## Table of Contents

+ [Summary](#summary)
+ [References](#references)
+ [How to use](#how-to-use)

## Summary

Simple Object detector app utilizing trained DETR models. The code in [object_detector_app.py](./object_detector_app.py) is copied from [rtdetrClass.py from niconielsen32](https://github.com/niconielsen32/DETR/blob/main/rtdetrClass.py).

## References

- [rtdetrClass.py from niconielsen32](https://github.com/niconielsen32/DETR/blob/main/rtdetrClass.py)
- [Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector](https://docs.ultralytics.com/models/rtdetr/)

## How to use

### Install pip packages

```sh
pip install -r requirements.txt

# if any errors occur upgrade pip packages in the following order
pip install --upgrade numpy
pip install --upgrade pillow requests pyyaml
pip install --upgrade matplotlib scipy pandas seaborn ultralytics supervision
pip install -r requirements.txt
```

### Run object detector application

```sh
# python object_detector_app.py
python3 object_detector_app.py
```