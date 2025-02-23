from detection_with_tracker.detection_with_tracker import DetectionManager
from detection_with_tracker.detection_with_tracker import Parameters

from display_functionality.app import WebServer

class MainManager:
    '''the main file which manages all the functionality'''
    def __init__(self):
        self.parameters = self.setParameters(rpicam=False)
        self.detection_manager = DetectionManager(self.parameters)

        self.server = WebServer(self)
        self.server.run()
    def setParameters(self,rpicam:bool=False):
        parameters = Parameters()
        parameters.set_model_paths(hef_path='model/yolov10n.hef', labels_path="detection_with_tracker/coco.txt")
        if not rpicam:
            parameters.set_input_video(input_video_path=Parameters.DEFAULT_VIDEO_PATHS[1])
        parameters.set_displaying(displayFrame=False,save_frame_debug=True)
        return parameters
    
    def start_process(self):
        '''Starts the entire program from button press'''
        # self.detection_manager.start_process()    
        self.server.set_start()
        self.running = True

        self.run()
    def stop_process(self):
        '''Stops the entire program from button press'''
        # self.detection_manager.stop_process()
        self.server.set_stop()
        self.running = False

    def run(self):
        while self.running:
            self.detection_manager.run_process()
            
            crash_id = self.detection_manager.crash_id
            crash_d = self.detection_manager.current_crash_d
            crash_t = self.detection_manager.crash_time

            self.server.update_data()

if __name__ == "__main__":
    main_manager = MainManager()