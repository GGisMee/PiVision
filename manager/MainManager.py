from detection_with_tracker.detection_with_tracker import DetectionManager
from detection_with_tracker.detection_with_tracker import Parameters

from display_functionality.app import WebServer

from buzzer.BuzzerManager import BuzzerManager

from voltage_tester.battery_voltage import VoltageTester
from time import time
import numpy as np

import threading

class MainManager:
    '''The main file which manages all the functionality'''
    def __init__(self):
        self.parameters = self.setParametersTesting(rpicam=False)
        self.detection_manager = DetectionManager(self.parameters)

        self.buzzer_manager = BuzzerManager()

        self.voltage_tester = VoltageTester()
        


        self.server = WebServer(self)
        self.server.run()

    def setParametersTesting(self,rpicam:bool=False):
        parameters = Parameters()
        parameters.set_model_paths(hef_path='model/yolov10n.hef', labels_path="detection_with_tracker/coco.txt")
        if not rpicam:
            crop_imaging = 'output/videos/crop_for_imaging_8.mp4'
            from_school = 'output/videos/no_fskola.mp4'
            from_home = 'output/videos/hemma1.mp4'
            
            parameters.set_input_video(input_video_path=crop_imaging)
        parameters.set_displaying(displayFrame=False,save_frame_debug=True)
        return parameters
    
    def setParametersFunctional(self):
        parameters = Parameters()
        parameters.set_model_paths(hef_path='model/yolov10n.hef', labels_path="detection_with_tracker/coco.txt")
        parameters.set_displaying(displayFrame=False,save_frame_debug=False)
        return parameters
    
    def start_process(self):
        '''Starts the entire program from button press'''
        # self.detection_manager.start_process()
        self.server.set_start()
        self.running = True
        self.process_thread = threading.Thread(target=self.run)
        self.process_thread.start()

        # self.run()
    def stop_process(self):
        '''Stops the entire program from button press'''
        # self.detection_manager.stop_process()
        self.running = False
        if self.process_thread.is_alive():
            self.process_thread.join()  # Ensure the thread stops cleanly

    def run(self):
        while self.running:
            if self.detection_manager.run_process():
                break # Either if the loop is finished running through or if it closed in some way 
            
            # vehicle
            if self.detection_manager.vehicle_detected:
                num_now = self.detection_manager.num_detections
                status = self.detection_manager.crash_status
                d_front = round(self.detection_manager.closest_front_distance,2)
                d_close = round(self.detection_manager.closest_d,2)
                latest_data = np.array(self.detection_manager.latest_data)
                
                # Används för att visa färgerna på bilarna i canvasen.
                ID_to_color = self.detection_manager.distance_estimator.ID_to_color
                # Används för att få rätt storlekar på bilarna i canvasen.
                ID_to_class = self.detection_manager.distance_estimator.data_corresponding_class
                self.buzzer_manager.check_play(status=status)

            else:
                self.buzzer_manager.check_play(status=0)
                continue
            
            self.voltage_procentage = self.voltage_tester.get_percentage_left()
            if not self.voltage_procentage:
                self.voltage_procentage = 50
            

            self.server.update_data(
                num_now=num_now,
                d_front=d_front,
                d_close=d_close,
                latest_data=latest_data,
                warning_status=status,
                ID_to_color = ID_to_color,
                ID_to_class = ID_to_class,
                voltage_procentage = self.voltage_procentage
            )

            



        else:
            # When the loop it run through after stop is pressed
            self.server.set_stop()


if __name__ == "__main__":
    print('Running from MainManager')
    main_manager = MainManager()