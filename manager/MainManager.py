from detection_with_tracker.detection_with_tracker import DetectionManager
from detection_with_tracker.detection_with_tracker import Parameters

from display_functionality.app import WebServer

import threading

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
        self.process_thread = threading.Thread(target=self.run, daemon=True)
        self.process_thread.start()

        self.run()
    def stop_process(self):
        '''Stops the entire program from button press'''
        # self.detection_manager.stop_process()
        self.server.set_stop()
        self.running = False
        if self.process_thread.is_alive():
            self.process_thread.join()  # Ensure the thread stops cleanly

    def run(self):
        while self.running:
            self.detection_manager.run_process()
            # vehicle
            if self.detection_manager.vehicle_detected:
                num_now = self.num_detections
                d_front = self.detection_manager.front_dist
                d_close = self.detection_manager.closest_distance

            else:
                num_now = '-'
                d_front = '-'
                d_close = '-'
            self.server.update_data(
                num_now=num_now,
                d_front=d_front,
                d_close=d_close,
            )

if __name__ == "__main__":
    print('Running from MainManager')
    main_manager = MainManager()