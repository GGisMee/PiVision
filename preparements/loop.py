import cv2
import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time

import load, runner

processing_fps = 5

grabber = load.ImageGrabber(load.ImageGrabber.getVideoPaths()[0])
model = YOLO(r'model/yolov8n.pt')
classes = list(model.names.values())
tracker = sv.ByteTrack(track_activation_threshold=0.4, lost_track_buffer=4 * processing_fps, minimum_matching_threshold=0.6, frame_rate=processing_fps, minimum_consecutive_frames=3)  # track_activation_threshold=0.4, lost_track_buffer=4, minimum_matching_threshold=0.8, frame_rate=10, minimum_consecutive_frames=0

plt.ion()
timeList = []
i: int = 0  # the index
detectionForImages = {}

while grabber.cap.isOpened():
    timeObj = time.time()
    i += 1
    
    if grabber.cap.get(cv2.CAP_PROP_POS_FRAMES) > 56:
        break

    image = grabber.getImg('Next')
    if isinstance(image, int):
        break
    
    # Makes sure that the fps are equal (close) to the processing_fps
    if not ((i % round(grabber.fps / processing_fps)) == 0):
        continue
    
    print(f'{i} done of {grabber.total_frames}, now: {grabber.cap.get(cv2.CAP_PROP_POS_FRAMES)}')

    detections = runner.predict(model, image, tracker)
    if not isinstance(detections, tuple):
        # detections.xyxy
        for obj in detections:
            x_min, y_min, x_max, y_max = obj[0]
            trackingID = obj[4]
            conf = obj[2]
            # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            label = f"ID: {trackingID}, Conf: {conf:.2f}"
            # cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if f'{trackingID}' not in detectionForImages:
                detectionForImages[f'{trackingID}'] = [obj[0]]
            else:
                detectionForImages[f'{trackingID}'].append(obj[0])

    # print('showing text')
    # cv2.putText(image, f'frame nr: {i}', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # print('showing image')
    # plt.imshow(image)
    # plt.title(f'Frame {i} Tracking')
    # plt.axis('off')
    # plt.show()
    # plt.pause(0.01)
    timeList.append(time.time()-timeObj)

print(timeList, np.mean(timeList))
print(detectionForImages)
# plt.ioff()

grabber.cap.release()
cv2.destroyAllWindows()
