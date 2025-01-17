'''Module is for testing and getting the functionality for the gymnasium project done'''
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO
import time

from load import ImageGrabber

def predict(model, image, tracker):
    '''Predicts how a model should act'''
    
    result:np.ndarray = model.predict(image)[0]
    
    if not result or result.boxes is None:
        return 1, ...
    
    detections = sv.Detections.from_ultralytics(result)
    detected_objs = tracker.update_with_detections(detections)

    return detected_objs


if __name__ == '__main__': 

    # inputs
    grabber = ImageGrabber(ImageGrabber.getVideoPaths()[0])
    image = grabber.getImg('Random')
    model_path: str = r'model/yolov8n.pt' 
    if not os.path.exists(model_path):
        exit("Model wasn't found")

    model = YOLO(model_path)

    tracker = sv.ByteTrack()

    detections = predict(model, image, tracker)
    if not isinstance(detections, tuple):
        # detections.xyxy
        for obj in detections:
            x_min, y_min, x_max, y_max = obj[0]
            print(f'ID {obj[4]} with {obj[0]}')
            trackingID = obj[4]
            conf = obj[2]
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 2)
            label = f"ID: {trackingID}, Conf: {conf:.2f}"
            cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.imshow(image)
    plt.show()
    pass