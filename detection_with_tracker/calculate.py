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


class DistanceEstimator:
    ID_TO_HEIGHTS = {2: 1.5, 5: 3, 7:2} # pairs up class_id:s to vehicles height.
    def __init__(self, parameters:'Parameters', class_names:list, wh:tuple[int]):
        '''A class to encapsule the distance estimation process
        
        Inputed args:
            parameters: Parameters class from detection_with_tracker =  with different parameters for the model, for example use_rpi
            class_names: list = names for the different classes.
            wh: tuple[int] = Should be be a tuple of width and height of the picture.
            
        Created args:
            self.data: dict = tracker_id : {'s':[distance1, distance2...], 't':[time1,time2,...]}
            self.data_corresponding_class: dict = tracker_id : class_id'''
        self.data: dict = {}
        self.data_corresponding_class: dict = {}

        self.class_names = class_names

        self.FOCAL_LENGTH = 248 # In pixels!
        self.list_length_cap = 20 # how many datapoints to store in the time distance lists..
        self.min_length_datapoints = 10
        self.start_time = None

        self.save_coming_distance = parameters.save_coming_distance
        self.display_coming_distance = parameters.display_coming_distance

        self.poly_fitter = PolyFitting(degree=1, weight_function_info={'min_weight': 0.1,'max_weight': 1,'scale_factor': 1,'decay_rate': 1,'mode': 'linear'}, saving= self.save_coming_distance, viewing=self.display_coming_distance)

        self.camera_angle = 0.5688062974 # angle. might produce poor composant estimates with other cameras. Made for raspberry pi
        self.frame_width = 640 # pixels in the aspect ratio which was converted to.
        self.distance_pixels = self.frame_width/(2*np.sin(self.camera_angle)) # Calculates the hypotinuse with trigonometry. More in calc_camera_values

    def add_detection(self, detections: sv.Detections):
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

            # adds the new data to the tracker_id
            self.data[tracker_id]['d'].append(d)
            self.data[tracker_id]['dx'].append(dx)
            self.data[tracker_id]['dy'].append(dy)
            self.data[tracker_id]['t'].append(time.time()-self.start_time)

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
    
    def check_crash(self):
        min_time = np.inf
        min_id = None
        latest_d = None

        for tracker_id in self.data.keys():
            if len(self.data[tracker_id]['t'])<self.min_length_datapoints:
                continue
            come_dx, come_dy, come_t = self.poly_fitter.update(self.data[tracker_id])

            # time = self.poly_fitter
            x_hit_interval = [-0.9, 0.9]
            y_hit_interval = [-2.3, 2.3]
            hit_check_x = (x_hit_interval[0] < come_dx) & (come_dx< x_hit_interval[1])
            hit_check_y = (y_hit_interval[0] < come_dy) & (come_dy< y_hit_interval[1])


            in_car = np.bitwise_and(hit_check_x, hit_check_y)
            if np.any(in_car):
                times_until_in_car = come_t[in_car]
                time_until_in_car = np.min(times_until_in_car)
                if time_until_in_car < min_time:
                    min_time = np.min(time_until_in_car)
                    latest_d = self.data[tracker_id]['d'][-1]
                    min_id = tracker_id
            
        return min_time, min_id, latest_d

    def get_front_dist(self):
        '''To get the distance to the car in front of you'''
        min_derivative = 1/8 # Om den befinner sig inom en där dx < min_derivative * dy
        closest_front_distance = np.inf
        for tracker_id in self.data.keys():
            if len(self.data[tracker_id]['t'])<self.min_length_datapoints:
                continue
            dx = self.data[tracker_id]['dx'][-1]
            dy = self.data[tracker_id]['dy'][-1]
            if (dx < min_derivative * dy) and dx<closest_front_distance:
                closest_front_distance = dy
        return closest_front_distance

    def get_closest_dist(self):
        '''To get the distance to the car closest to you'''
        closest_d = np.inf
        for tracker_id in self.data.keys():        
            closest_d_id = self.data[tracker_id]['d'][-1]
            if closest_d > closest_d_id:
                closest_d = closest_d_id
        return closest_d

def derive(y):
    return np.insert((y[1:]-y[:-1]),0,0)
    
def viewData(distance:np.ndarray, speed: np.ndarray, acceleration:np.ndarray):
    plt.yticks(np.arange(-1,7,0.2))
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.plot(distance, color = 'red')
    plt.plot(speed, color = 'blue')
    plt.plot(acceleration, color = 'green')
    plt.show()

def getDistance(xyxys:dict):
    distData = {}
    W_actual:float = 1.8
    focal:int=248
    for id in xyxys.keys():
        #print(f'ID: {id}')
        for xyxy in xyxys[id]:
            W_pixels = abs(xyxy[0]-xyxy[2])
            Distance = (W_actual*focal)/W_pixels

            if id not in distData:
                distData[id] = [Distance]
            else:
                distData[id].append(Distance)
        distData[id] = np.array(distData[id])
    return distData

def get_speed_and_acceleration(distData:dict):
    speedData = {}
    accData = {}
    for id in distData.keys():
        speed = derive(distData[id])
        acceleration = derive(speed)
        if id not in speedData:
                speedData[id] = speed
        else:
            speedData[id].append(speed)
        if id not in accData:
                accData[id] = acceleration
        else:
            accData[id].append(acceleration)
    return speedData, accData
    
if __name__ == '__main__':
    pass