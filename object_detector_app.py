import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import argparse  

import supervision as sv
from ultralytics import RTDETR


class ObjectDetection:
    def __init__(self, capture_index, model_choice):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        if model_choice == 'nano_yolov8':
            self.model = YOLO("yolov8n.pt")  
        elif model_choice == 'small_yolov8':
            self.model = YOLO("yolov8s.pt")
        elif model_choice == 'nano_yolov11':
            self.model = YOLO("yolo11n.pt")  
        elif model_choice == 'small_yolov11':
            self.model = YOLO("yolo11s.pt")  
        elif model_choice == 'large_detr':
            self.model = RTDETR("rtdetr-l.pt")  
        elif model_choice == 'extra_large_detr':
            self.model = RTDETR("rtdetr-x.pt")  
        else:
            raise ValueError("Invalid model choice. Choose from 'nano_yolov8', 'small_yolov8', 'large_detr', 'extra_large_detr'.")

        self.CLASS_NAMES_DICT = self.model.model.names
        print(self.CLASS_NAMES_DICT)

        self.box_annotator = sv.BoxAnnotator(color=sv.Color(255, 0, 0), thickness=3)  
        self.label_annotator = sv.LabelAnnotator()

    def plot_bboxes(self, results, frame):
        
        boxes = results[0].boxes.cpu().numpy()
        class_id = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy
        
        class_id = class_id.astype(np.int32)
              
        detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=class_id,
                    )
        
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id[i]]} {conf[i]:0.2f}" for i in range(len(class_id))]

        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start_time = time()
            ret, frame = cap.read()

            results = self.model.predict(frame, verbose=False)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('Object Detector App', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select pre-trained model for object detection.')
    parser.add_argument("--model", type=str, required=True, choices=['nano_yolov8', 'small_yolov8', 'nano_yolov11', 'small_yolov11', 'large_detr', 'extra_large_detr'], help='Choose a model: nano_yolov8, small_yolov8, nano_yolov11, small_yolov11, large_detr, extra_large_detr')
    args = parser.parse_args()

    
    detector = ObjectDetection(capture_index=0, model_choice=args.model)
    detector()
