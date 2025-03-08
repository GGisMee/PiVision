import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from collections import deque
import time
from detection_with_tracker.fit_to_points import PolyFitting
import json
from typing import TYPE_CHECKING

# Ensures that type anotations work without circular imports
if TYPE_CHECKING:
    from detection_with_tracker.detection_with_tracker import Parameters  


class CrashCalculater:
    ID_TO_HEIGHTS = {2: 1.5, 5: 3, 7:3} # pairs up class_id:s to vehicles height.
    def __init__(self, parameters:'Parameters', class_names:list, wh:tuple[int]):
        '''A class to encapsule the distance estimation process
        
        Inputed args:
            parameters: Parameters class from detection_with_tracker =  with different parameters for the model, for example use_rpi
            class_names: list = names for the different classes.
            wh: tuple[int] = Should be be a tuple of width and height of the picture.
            
        Created args:
            self.data: dict = tracker_id : {'s':[distance1, distance2...], 't':[time1,time2,...], 'last_seen_timestamp':time.time()}
            self.data_corresponding_class: dict = tracker_id : class_id'''
        self.data: dict = {}
        self.data_corresponding_class: dict = {}

        self.class_names = class_names

        self.FOCAL_LENGTH = 248 # In pixels!
        self.multiplier = 1 # just to increase the size of the FOCAL_LENGTH to ensure that the picture gets more correct. Must be an issue somewhere.
        self.FOCAL_LENGTH*=self.multiplier
        self.list_length_cap = parameters.datapoint_cap
        self.min_length_datapoints = 10
        self.start_time = None

        # the time until the car is forgotten. Reason is to ensure that the cars doesn't clip stuck on screen.
        self.time_to_forget = 2 # seconds

        self.ID_to_color = {}


        self.save_coming_distance = parameters.save_coming_distance
        self.display_coming_distance = parameters.display_coming_distance

        self.poly_fitter = PolyFitting(degree=1, weight_function_info={'min_weight': 0.1,'max_weight': 1,'scale_factor': 1,'decay_rate': 1,'mode': 'linear'}, saving= self.save_coming_distance, viewing=self.display_coming_distance)

        self.camera_angle = 0.5688062974 # angle. might produce poor composant estimates with other cameras. Made for raspberry pi
        self.frame_width = 640 # pixels in the aspect ratio which was converted to.
        self.distance_pixels = self.frame_width/(2*np.sin(self.camera_angle)) # Calculates the hypotinuse with trigonometry. More in calc_camera_values

    def add_detection(self, detections: sv.Detections, frame:np.ndarray):
        '''Adds the detections to the data dictionary, which is a dictionary keeping track of the distances and the time to each car.'''
        if not self.start_time:
            self.start_time = time.time()
        for i,tracker_id in enumerate(detections.tracker_id):
            # Gets the distance corresponding the tracker_id
            d,dx,dy = self._get_distance(detections.xyxy[i], detections.class_id[i])
            # create key_value pair if it doesn't exist, otherwise append
            if tracker_id not in self.data.keys():
                # create the specific data holders for the tracker_id
                # I use a deque here to limit the amount of variables in the list. 
                # This makes sure that the num calculations stay low and only relative data is used
                distance_deque = deque(maxlen=self.list_length_cap)
                dx_deque = deque(maxlen=self.list_length_cap)
                dy_deque = deque(maxlen=self.list_length_cap)
                time_deque = deque(maxlen=self.list_length_cap)
                self.data[tracker_id] = {'d':distance_deque, 'dx':dx_deque, 'dy':dy_deque, 't':time_deque}

                # adds the class of the tracker_id
                self.data_corresponding_class[tracker_id] = detections.class_id[i]

                # adds the color for the newly seen car to dict ID_to_color
                self._match_color(xyxy=detections.xyxy[i], frame=frame, tracker_id=tracker_id)

            # adds the new data to the tracker_id
            self.data[tracker_id]['d'].append(d)
            self.data[tracker_id]['dx'].append(dx)
            self.data[tracker_id]['dy'].append(dy)
            self.data[tracker_id]['t'].append(time.time()-self.start_time)
        self._remove_old_trackers(detections)
                
    def _remove_old_trackers(self,detections):
        old_tracker_ids = set(self.data.keys()).difference(detections.tracker_id)
        for id in old_tracker_ids:
            last_timestamp = self.data[id]['t'][-1]+self.start_time
            if self.time_to_forget < time.time()-last_timestamp:
                self.data.pop(id)
                self.ID_to_color.pop(id)

    def _match_color(self, xyxy:np.ndarray,frame:np.ndarray, tracker_id:int):
        x1,y1,x2,y2 = xyxy.astype(np.int16)
        roi = frame[y1:y2, x1:x2]
        mean_color = np.mean(roi, axis = (0,1))
        self.ID_to_color[tracker_id] = mean_color
        

    def _get_distance(self, xyxy, class_id):
        '''Determines the distance to a vehicle.
        
        Parameters:
            xyxy: A xyxy box array in the frame
            class_id: The corresponding class_id, i.e. which type of vehicle it is.'''
        def get_real_world_distance() -> float:
            h_pixels:int = xyxy[3] - xyxy[1]  # Bounding box height in pixels
            w_pixels:int = xyxy[2] - xyxy[0]  # Bounding box width in pixels

            real_height = self.ID_TO_HEIGHTS[class_id]  # Known real-world height
            # Depth estimation (distance along Z-axis)
            d = real_height * self.FOCAL_LENGTH / h_pixels
            return d

        def get_composites(d:float):
            distance_from_center_x = (xyxy[0]+xyxy[2])/2- self.frame_width/2
            dx = d/self.distance_pixels*(distance_from_center_x) # for aspect ratio of 1920x1080
            # dy = d*np.sqrt(1-(distance_from_center_x**2)/(self.distance_pixels**2))
            dy = np.sqrt(d**2-dx**2) # simple use of pythagoras theorem, equal to the one above, but less calculations
            return dx,dy
    
        d = get_real_world_distance()
        dx, dy = get_composites(d)

        return (d,dx , dy)  # 3D vector from camera to bbox center

    def get_display_labels(self, sv_detections: sv.Detections):
        '''Get the labels which will then be displayed for each of the cars.'''
        labels = []
        for tracker_id in sv_detections.tracker_id:
            class_id = self.data_corresponding_class[tracker_id]
             
            # gets the latest distance associated with tracker_id. Latest, hence -1
            d = self.data[tracker_id]['d'][-1] 
            dx = self.data[tracker_id]['dx'][-1]

            label = f"#{tracker_id}, {self.class_names[class_id]}, {d:.2f}, {dx:.2f}"
            labels.append(label)
        return labels
    
    def _check_crash(self,x_coeffs:np.ndarray,y_coeffs:np.ndarray,  min_time:float, highest_time:float):
        '''Checks if a crash is coming and the time until this crash happens. Run through self.dataloop
        
        Input:
            x_coeffs = A list of coefficients for a regression created polynomial. For the x values
            y_coeffs = A list of coefficients for a regression created polynomial. For the y values
            min_time = time value until crash. Will get updated through this function
            highest_time = The latest time datapoint. Useful to determine where the new test datapoints should start.

        Returns:
            new_min_time: lowest time until crash. 
            x_coeffs: coefficients of the regression model for the x values
            y_coeffs: coefficients of the regression model for the y values
            '''
        
        t_boundry_scope: list[float] = [0.1,5]
        t_boundries = [highest_time+t_boundry_scope[0], highest_time+t_boundry_scope[1]]

        come_t:np.ndarray = np.linspace(t_boundries[0], t_boundries[1], num = 10)
        come_dx:np.ndarray[float] = np.polyval(x_coeffs, come_t)
        come_dy:np.ndarray[float] = np.polyval(y_coeffs, come_t)

        # The interval where its in the car. By this I say that the car is 1.8 m in width and 4.6 m in length.
        x_hit_interval = [-0.9, 0.9] 
        y_hit_interval = [-2.3, 2.3]

        hit_check_x = (x_hit_interval[0] < come_dx) & (come_dx< x_hit_interval[1])
        hit_check_y = (y_hit_interval[0] < come_dy) & (come_dy< y_hit_interval[1])

        in_car = np.bitwise_and(hit_check_x, hit_check_y)
        if not np.any(in_car): # if it is not going to get into the car, then don't care about it.
            return None

        times_until_in_car = come_t[in_car] # gets the timepoints when its in the car.
        time_until_in_car = np.min(times_until_in_car) # takes out the lowest value
        if time_until_in_car < min_time: # if smaller
            new_min_time = np.min(time_until_in_car)    
            return new_min_time
        # If the new value for crash is higher then the last.
        return None
        
    def dataloop(self):
        '''A loop which calculates all the different data which will then be displayed'''
        
        closest_front_distance = np.inf
        closest_d = np.inf
        min_time = np.inf
        latest_data = [] # a dataset for the display of the vehicles. Includes coming position in one second and the current position
        # [id, dx,dy,vx,vy]

        for tracker_id in self.data.keys():
            t = np.array(self.data[tracker_id]['t'])
            dy = np.array(self.data[tracker_id]['dy'])
            dx = np.array(self.data[tracker_id]['dx'])

            if closest_front_distance_new_maybe := self._get_front_dist(tracker_id, closest_front_distance):
                closest_front_distance = closest_front_distance_new_maybe

            if closest_d_new := self._get_closest_dist(tracker_id, closest_d):
                closest_d = closest_d_new

            if len(t) < 20: # a cap to ensure that the fitted regression won't be inaccurate
                latest_data.append([tracker_id, dx[-1], dy[-1], 0, 0])
                continue

            # Gets the coefficiants for polynomial model from the data which already exists
            x_coeffs:np.ndarray = self.poly_fitter.get_regression_model(dx, t)
            y_coeffs:np.ndarray = self.poly_fitter.get_regression_model(dy, t)

            # Computes vectors where the cars might go next.
            vx, vy = self.get_coming_vector(t[-1], dx[-1], dy[-1], x_coeffs, y_coeffs)
            latest_data.append([tracker_id, dx[-1], dy[-1], vx,vy])

            new_min_time = self._check_crash(x_coeffs, y_coeffs, min_time=min_time, highest_time=max(t))
            if new_min_time:
                min_time = new_min_time


        
        status = self._get_crash_status(min_time)

        return closest_front_distance, closest_d, status, latest_data
            
    def get_coming_vector(self, latest_t, latest_dx, latest_dy, coeffs_x, coeffs_y):
        new_t = latest_t + 1
        new_x = self.poly_fitter.get_values_from_model(coeffs_x, [new_t])[0]
        new_y = self.poly_fitter.get_values_from_model(coeffs_y, [new_t])[0]

        vx = new_x-latest_dx
        vy = new_y-latest_dy
        return (vx,vy)

    def _get_front_dist(self, tracker_id, closest_front_distance):
        '''To get the distance to the car in front of you'''
        min_derivative = 1/8 # If it exists within a triangle in front of the car. This adds a bit of extra allowance for what is actually in front
            # stop för att se till att allt för nya bilar inte registreras
        if len(self.data[tracker_id]['t'])<self.min_length_datapoints:
            return None
        dx = self.data[tracker_id]['dx'][-1]
        dy = self.data[tracker_id]['dy'][-1]
        if (abs(dx) < min_derivative * dy) and dx<closest_front_distance:
            return dy

    def _get_closest_dist(self, tracker_id, closest_d):
        '''To get the distance to the car closest to you'''     
        closest_d_for_id = self.data[tracker_id]['d'][-1]
        if closest_d > closest_d_for_id:
            return closest_d_for_id

    def _get_crash_status(self, min_time):
        '''Depending on the time until crash a status is returned, which is a value between 0 and 9,
        where 0 is the safest'''
        if min_time >= 3:
            return 0
        status = round(min_time*3)
        status = 9 if status > 9 else status
        return status
    
    def _get_datapoints(self):
        '''Returns current datapoints, along with vector showing direction'''
        datapoints = []
        for tracker_id in self.data.keys():
            dx = self.data[tracker_id]['dx'][-1]
            dy = self.data[tracker_id]['dy'][-1]
            vx:float
            vy:float
            datapoints.append([dx,dy,vx,vy])



if __name__ == '__main__':
    pass