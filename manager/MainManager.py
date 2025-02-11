from detection_with_tracker.detection_with_tracker import DataManager
from detection_with_tracker.detection_with_tracker import Parameters

# python -m manager.MainManager

def setParameters():
    parameters = Parameters()
    parameters.set_model_paths(hef_path='model/yolov10n.hef', labels_path="detection_with_tracker/coco.txt")
    parameters.set_input_video(input_video_path=Parameters.DEFAULT_VIDEO_PATHS[1])
    parameters.set_displaying(displayFrame=False,save_frame_debug=True)
    return parameters


parameters = setParameters()
data_manager = DataManager(parameters)
while True:
    data_manager.run()