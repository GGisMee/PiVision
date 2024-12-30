'''Module is for testing and getting the functionality for the gymnasium project done'''
import cv2

import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO

from load import ImageGrabber

MIN_THRESHOLD = 0.4

# inputs
grabber = ImageGrabber(ImageGrabber.getVideoPaths()[0])
image = grabber.getImg('Random')

model = YOLO(r'model\yolov8n.pt')


tracker = sv.ByteTrack(track_activation_threshold=0.4, lost_track_buffer=4, minimum_matching_threshold=0.8, frame_rate=10, minimum_consecutive_frames=3)


def predict(model, image):
    '''Predicts how a model should act'''
    
    result:np.ndarray = model.predict(image)[0]
    
    if not result or result.boxes is None:
        return 1, ...
    
    detections = sv.Detections.from_ultralytics(result)
    detected_objs = tracker.update_with_detections(detections)
    # Update BYTETracker
    # tracked_objects = tracker.update(detections, image)

    pass





    # for obj in tracked_objects:
    #     tlwh = obj.tlwh
    #     track_id = obj.track_id
    #     x1, y1, w, h = map(int, tlwh)
    #     x2, y2 = x1 + w, y1 + h

    #     # Draw bounding box and ID
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(image, 
    #                 f"ID {track_id}", (x1, y1 - 10), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 
    #                 0.5, (0, 255, 0), 2)

    # Show image
    return image

    # boxes = result.boxes  # Boxes object for bounding box outputs
    # # Iterate through the detected boxes

    # xyxy = []
    # confidenceScores = []
    # ids = []
    # names = []
    # for box in boxes:
    #     if conf := box.conf.item() < box_threshold:
    #         continue
    #     elif 
    #     xyxy.append(box.xyxy.numpy())  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    #     confidenceScores.append(conf)  # Confidence score of the detection
    #     ids.append(box.cls.item())  # Class ID of the detected object
    #     names.append(model.names[int(box.cls)])  # Human-readable class name
    # xyxy, confidenceScores, ids, names = np.array(xyxy), np.array(confidenceScores), np.array(ids), np.array(names)
    #return result, xyxy, confidenceScores, ids, names


# result, xyxy, confidenceScores, ids, names = predict(model, image)

# for i,xyxy_el in enumerate(xyxy):
#     print(xyxy_el, confidenceScores[i])

# result.show()  # display to screen

new_image = predict(model, image)
cv2.imshow('Tracking', new_image)