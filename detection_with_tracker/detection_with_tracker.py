"""Example module for Hailo Detection + ByteTrack + Supervision."""

import supervision as sv
import numpy as np
from tqdm import tqdm
import cv2
import queue
import sys
import os
from typing import Dict, List, Tuple
import threading
import time
from picamera2 import Picamera2
from rich.console import Console
from rich.live import Live
from rich.text import Text
import psutil
from collections import deque

from detection_with_tracker.calculate import DistanceEstimator


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.hailo_utils import HailoAsyncInference


class Parameters:
    '''A container for the variables in the detection algoritm
    
    Optional:
        set_model_paths
        set_model_info
        set_input_video
        set_max_fps
        set_displaying
    '''

    DEFAULT_VIDEO_PATHS = ['resources/videos/detection0.mp4', 'resources/videos/close616.mov', 'resources/videos/kaggle_bundle/00067cfb-e535423e.mov']

    def __init__(self):
        # Default values if not changes
        self.use_rpi = True # if not set_input_video
        self.create_output_video: bool = None # until changed
        self.score_threshold = 0.5
        self.displayFrame = True
        self.hef_path = 'model/yolo10n.hef'
        self.labels_path = "detection_with_tracker/coco.txt"
        
        self.save_frame_debug = False
        self.max_fps = None


    def _test_existance(self,paths:list[str]):
        '''Tests paths if they exist'''
        for specific_path in paths:
            if not os.path.exists(specific_path):
                print(f"{specific_path} was not found")
                assert FileNotFoundError(f'File of path {specific_path} does not exist')

    def set_model_paths(self, hef_path:str, labels_path:str=None):
        '''Sets the paths for the model'''
        if not labels_path:
            labels_path = os.getcwd()+"/coco.txt"

        self._test_existance([hef_path, labels_path])
        

        self.hef_path = hef_path
        self.labels_path = labels_path

    def set_max_fps(self, max_fps:int=10):
        self.max_fps = max_fps

    def set_model_info(self, score_threshold:float = 0.5):
        '''Sets the parameters for the model, that is how the model should act'''
        self.score_threshold = score_threshold

    def set_input_video(self, input_video_path:str, focal_length:int = 35):
        '''If the raspberry pi shouldn't be used and the input should come from a video instead'''
        self.use_rpi = False
        self._test_existance([input_video_path])
        self.input_video_path = input_video_path

    def create_output(self, output_video_path:str):
        '''If a output video should be saved'''
        self._test_existance([output_video_path])
        self.create_output_video = True
        self.output_video_path:str= output_video_path

    def set_displaying(self, displayFrame:bool = False, save_frame_debug: bool = False, display_coming_distance:bool = False, save_coming_distance:bool=False):
        '''Used to give an annotated frame of the cars with their boundry boxes and other info
        WARNING: displaying and especially saving is really time consuming'''
        self.displayFrame = displayFrame
        self.save_frame_debug = save_frame_debug
        self.save_coming_distance = save_coming_distance
        self.display_coming_distance = display_coming_distance

class FrameGrabber:
    '''A class to handle the frame creation process'''
    def __init__(self, parameters:Parameters):
        self.use_rpi = parameters.use_rpi
        self.running = True
        self.index = 0

        if self.use_rpi:
            self.camera = Picamera2()
            # Here we configure the camera to take in a size of 1920x1080. We also make sure that it is in RGB and that it is for video capture
            # We use size of 1920x1080 because since it is a good fit between a large size and a less detailed one. It also matches the cameras aspect ratio meaning that nothing is cropped out of the picture.  
            camera_config = self.camera.create_video_configuration(main={"size": (1920, 1080), 'format': 'RGB888'})
            
            self.camera.configure(camera_config)
            self.camera.start()

        else:
            self.frame_generator = sv.get_video_frames_generator(source_path=parameters.input_video_path)
            self.video_info = sv.VideoInfo.from_video_path(video_path=parameters.input_video_path)
    
    def get_wh_set_generator(self):
        if self.use_rpi:
            video_w, video_h = 1920, 1080
        else:
            video_w, video_h = self.video_info.resolution_wh
        return video_w, video_h

    def get_frame(self):
        self.index += 1
        # print(f'{self.index} - {self.video_info.total_frames}')
        if self.use_rpi:
            frame = self.camera.capture_array()

            #! Temporary code to stop capturing:
            if self.index == 500:
                return True
        else:
            if self.index == self.video_info.total_frames:
                # To check that the video has been run through
                return True
            frame = next(self.frame_generator)
        return frame
     
class FrameNumberHandler:
    def __init__(self, max_fps: int = None):
        self.current_frame = 0
        self.fps = None
        self.start_time = None
        self.max_fps = max_fps
    def update_frame(self):
        self.current_frame += 1
        if not self.start_time:
            self.start_time = time.time()
    def update_fps(self):
        new_time = time.time()-self.start_time
        self.fps = 1 / new_time if new_time > 0 else 0
        if self.fps>self.max_fps:
            time.sleep(1/self.max_fps-1/self.fps)
            new_time = time.time()-self.start_time
            self.fps = 1 / new_time if new_time > 0 else 0
        self.start_time = time.time()
        
class Displayer:
    '''A class to handle what is displayed both with cv2 tools and terminal'''
    def __init__(self, parameters: Parameters):
        self.use_rpi: bool = parameters.use_rpi
        self.displayFrame: bool = parameters.displayFrame
        self.setup_rich_debug()
        self.save_frame_debug: bool = parameters.save_frame_debug

    def display_frame(self,frame):
        """
        Display the frame in a window using OpenCV.
        Press 'q' to exit the display window.
        """
        if self.save_frame_debug:
            self.save_img(frame)
        if not self.displayFrame:
            return True

        cv2.imshow("Object Detection", frame)  # Display the frame in a window titled "Object Detection"

        # Wait for a key press for 1 ms. Press 'q' to quit the display.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False  # Signal to stop the program
        return True
    
    def update_detection_procentage(self, detections:bool):
        self.detection_procentage.append(detections)

    def setup_rich_debug(self):
        # rich debug info:
        self.console = Console()
        self.cap = 20
        self.detection_procentage = deque([False]*self.cap, maxlen=self.cap)
        self.live = Live(console=self.console, auto_refresh=True)
        self.live.start()

    def display_text(self, frame_number_handler: FrameNumberHandler):

        frame_count = frame_number_handler.current_frame
        fps = frame_number_handler.fps

        # Create rich text to display in the live view
        live_text = Text(f"Frame: {frame_count}, FPS: {fps:.2f}, CPU: {psutil.cpu_percent()}%, Procentage {round(sum(self.detection_procentage)/self.cap*100)}%", style="bold green")

        # Update the live view with the latest frame info
        self.live.update(live_text)

    def stop_displaying(self):
        self.live.stop()

    def save_img(self,frame:np.ndarray):
        cv2.imwrite(filename='output/showed_img.png', img=frame)

class DetectionManager:
    def __init__(self, parameters:Parameters):
        self.parameters = parameters

        self.input_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()

        # set up the hailo inference functionality
        self.hailo_inference = HailoAsyncInference(
            hef_path= self.parameters.hef_path,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
        )
        self.model_h, self.model_w, _ = self.hailo_inference.get_input_shape() # 640x640 for hailo10n


        # Initialize components for video processing
        self.box_annotator = sv.RoundBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.tracker = sv.ByteTrack()
        # start, end = sv.Point(x=0, y=1080), sv.Point(x=3840, y=1080)
        # line_zone = sv.LineZone(start=start, end=end)

        # Load class names from the labels file
        with open(self.parameters.labels_path, "r", encoding="utf-8") as f:
            self.class_names: List[str] = f.read().splitlines()

        # Start the asynchronous inference in a separate thread
        self.inference_thread: threading.Thread = threading.Thread(target=self.hailo_inference.run)
        self.inference_thread.start()

        #* setup custom classes
        # start the framegrabber:
        self.framegrabber = FrameGrabber(self.parameters)
        self.frame_w, self.frame_h = self.framegrabber.get_wh_set_generator()

        self.distance_estimator = DistanceEstimator(self.parameters, self.class_names, (self.frame_w, self.frame_h))
        self.displayer = Displayer(self.parameters)
        self.frame_number_handler = FrameNumberHandler(self.parameters.max_fps)

    def run_process(self):
        '''Runs through a loop taking the frame, running it through the ai, getting the detections, tracking them, updating distance and checking for danger'''
        #* Get frame
        self.frame_number_handler.update_frame()
        # displayer.start_timer()
        frame=self.framegrabber.get_frame()
        if isinstance(frame, bool):
            return 1 # If last frame is reached
        
        # Preprocess the frame
        preprocessed_frame: np.ndarray = self._preprocess_frame(frame)

        #* hailo setup
        # Put the frame into the input queue for inference
        self.input_queue.put([preprocessed_frame])

        # Get the inference result from the output queue
        results: List[np.ndarray]
        _, results = self.output_queue.get()

        # Deals with the expanded results from hailort versions < 4.19.0
        if len(results) == 1:
            results = results[0]

        #* Extract detections from the inference results
        detections: Dict[str, np.ndarray] = self._extract_detections(
            results, self.model_h, self.model_w, self.parameters.score_threshold
        )

        #* view detections
        self.displayer.update_detection_procentage(bool(len(detections['class_id'])))
        self.frame_number_handler.update_fps()
        self.displayer.display_text(frame_number_handler=self.frame_number_handler)

        #* process detections
        if len(detections['class_id']) == 0:
            return False


        #* Postprocess the detections 
        sv_detections = self._process_detections(detections=detections)

        # if (self.frame_number_handler.current_frame % 50) == 0:
        #     timetester.print_execution_times()

        # if displaying is done in any way
        if self.parameters.displayFrame or self.parameters.save_frame_debug:
            #* Annotate the features such as bounding boxes and distance in title to frame
            annotated_labeled_frame = self._annotate_frame(frame=frame,sv_detections=sv_detections)
        
            #* Displays the frame
            if not self.displayer.display_frame(annotated_labeled_frame):
                return 1 # if q is pressed in the displayer
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the frame to match the model's input size."""

        # checks if already right size.
        if self.model_h != self.frame_h or self.model_w != self.frame_w:
            target_w, target_h = self.model_w, self.model_h
            input_w, input_h = self.frame_w, self.frame_h

            # Takes min and applies to both to keep aspect ratio
            scale = min(target_w / input_w, target_h / input_h)  
            new_w, new_h = int(input_w * scale), int(input_h * scale)

            # resizes to new size.
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Computes padding
            pad_top = (target_h - new_h) // 2 # How much to add top
            pad_bottom = target_h - new_h - pad_top # How much to add bottom
            pad_left = (target_w - new_w) // 2 # How much to add left
            pad_right = target_w - new_w - pad_left # How much to add to the right

            # Adds padding
            padded_image = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
            # returns
            return padded_image
        return frame

    def _extract_detections(
        self, hailo_output: List[np.ndarray], h: int, w: int, threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """Extract detections from the HailoRT-postprocess output."""
        xyxy: List[np.ndarray] = []
        confidence: List[float] = []
        class_id: List[int] = []
        num_detections: int = 0

        for i, detections in enumerate(hailo_output):
            if len(detections) == 0:
                continue
            for detection in detections:
                bbox, score = detection[:4], detection[4]

                if score < threshold:
                    continue

                # Convert bbox to xyxy absolute pixel values
                bbox[0], bbox[1], bbox[2], bbox[3] = (
                    bbox[1] * w,
                    bbox[0] * h,
                    bbox[3] * w,
                    bbox[2] * h,
                )
                if i in [2,5,7]:
                    xyxy.append(bbox)
                    confidence.append(score)
                    class_id.append(i)
                    num_detections += 1

        return {
            "xyxy": np.array(xyxy),
            "confidence": np.array(confidence),
            "class_id": np.array(class_id),
            "num_detections": num_detections,
        }

    def _process_detections(
        self,
        detections: Dict[str, np.ndarray],
        ) -> np.ndarray:

        sv_detections = sv.Detections(
            xyxy=detections["xyxy"],
            confidence=detections["confidence"],
            class_id=detections["class_id"]
        )

        #* Update detections with tracking information
        sv_detections = self.tracker.update_with_detections(sv_detections)

        self.distance_estimator.add_detection(sv_detections)
        self.crash_time, self.crash_id, self.current_crash_d = self.distance_estimator.check_crash()
        self.front_dist = self.distance_estimator.get_front_dist()
        self.closest_distance = self.distance_estimator.get_closest_dist()
        return sv_detections
        
    def _annotate_frame(self,frame: np.ndarray,sv_detections:sv.Detections) -> np.ndarray:
        """Postprocess the detections by annotating the frame with bounding boxes and labels."""
        
        # only if it has to be displayed, otherwise not necessary
        # gets labels for displaying
        labels: List[str] = self.distance_estimator.get_display_labels(sv_detections)

        # Annotate objects with bounding boxes
        annotated_frame: np.ndarray = self.box_annotator.annotate(
            scene=frame.copy(), detections=sv_detections
        )
        # Annotate objects with labels
        annotated_labeled_frame: np.ndarray = self.label_annotator.annotate(
            scene=annotated_frame, detections=sv_detections, labels=labels
        )

        return annotated_labeled_frame

def setParameters():
    parameters = Parameters()
    parameters.set_model_paths(hef_path='model/yolov10n.hef', labels_path="detection_with_tracker/coco.txt")
    parameters.set_input_video(input_video_path=Parameters.DEFAULT_VIDEO_PATHS[1])
    parameters.set_displaying(displayFrame=False,save_frame_debug=True, display_coming_distance=False, save_coming_distance=False)
    parameters.set_max_fps()
    return parameters
    
if __name__ == "__main__":
    print('Running from detection_with_tracker')
    parameters = setParameters()
    data_manager = DetectionManager(parameters)
    while True:
        data_manager.run_process()

        
    