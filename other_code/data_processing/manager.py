import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import hailo
from basic_pipelines.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from basic_pipelines.detection_pipeline import GStreamerDetectionApp



def start_ai():
    user_data = app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()


def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"



    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # printing the detections.
    printDetections(detections, string_to_print)
    
    # viewFrame(user_data, detection_count, frame)
    
    

    return Gst.PadProbeReturn.OK

def printDetections(detections, string_to_print:str):
        '''Adds labels which are then printed'''
        detection_count = 0
        for detection in detections:
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            if label == "car" or label == 'truck' or label == 'buss':
                string_to_print += f"Detection: {label} {confidence:.2f}\n"
        return string_to_print


class DetectionManager:
    def __init__(self):
        ...
        #tracker = sv.ByteTrack(track_activation_threshold=0.4, lost_track_buffer=4 * processing_fps, minimum_matching_threshold=0.6, frame_rate=processing_fps, minimum_consecutive_frames=3)  # track_activation_threshold=0.4, lost_track_buffer=4, minimum_matching_threshold=0.8, frame_rate=10, minimum_consecutive_frames=0



    

if __name__ == '__main__':
    dm = DetectionManager()
    start_ai()