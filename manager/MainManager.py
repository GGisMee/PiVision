from detection_with_tracker.detection_with_tracker import DataManager
from detection_with_tracker.detection_with_tracker import Parameters

from display_functionality.app import RealTimeServer

# python -m manager.MainManager



class MainManager:
    def __init__(self):
        data_manager = DataManager()
        server = RealTimeServer()
        self.parameters = self.setParameters(rpicam=False)
    def setParameters(self,rpicam:bool=False):
        parameters = Parameters()
        parameters.set_model_paths(hef_path='model/yolov10n.hef', labels_path="detection_with_tracker/coco.txt")
        if not rpicam:
            parameters.set_input_video(input_video_path=Parameters.DEFAULT_VIDEO_PATHS[1])
        parameters.set_displaying(displayFrame=False,save_frame_debug=True)
        return parameters
    
    
