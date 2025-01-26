import numpy as np
import matplotlib.pyplot as plt
import supervision as sv

class DistanceEstimater:
    ID_TO_HEIGHTS = {2: 1.5, 5: 3, 7:2} # pairs up class_id:s to vehicles height.

    def __init__(self, parameters, class_names:list):
        '''A class to encapsule the distance estimation process
        
        Inputed args:
            parameters: Parameters class from detection_with_tracker =  with different parameters for the model, for example use_rpi
            class_names: list = names for the different classes.

        Created args:
            self.data: dict = tracker_id : [distance1, distance2...]
            self.data_corresponding_class: dict = tracker_id : class_id'''
        self.data: dict = {}
        self.data_corresponding_class: dict = {}

        self.class_names = class_names

        self.FOCAL_LENGTH = 248
     
    def add_detection(self, detections: sv.Detections):
        '''Adds the detections to the data dictionary, which is a dictionary keeping track of the distances to each car.'''
        pass
        for i,tracker_id in enumerate(detections.tracker_id):
            # Gets the distance corresponding the tracker_id
            distance = self._get_distance(detections.xyxy[i], detections.class_id[i])

            # create key_value pair if it doesn't exist, otherwise append
            if tracker_id not in self.data.keys():
                self.data[tracker_id] = [distance]
                # adds the class of the tracker_id
                self.data_corresponding_class[tracker_id] = detections.class_id[i]
            else:
                self.data[tracker_id].append(distance)

    def _get_distance(self, xyxy, class_id):
        '''Gets the distance from a xyxy box'''
        h_pixels = xyxy[3]-xyxy[1]
        real_height = self.ID_TO_HEIGHTS[class_id]
        # The distance estimation itself
        Distance = real_height*self.FOCAL_LENGTH/h_pixels
        return Distance
    
    def get_display_labels(self, sv_detections: sv.Detections):
        labels = []
        for tracker_id in sv_detections.tracker_id:
            class_id = self.data_corresponding_class[tracker_id]
             
            # gets the latest distance associated with tracker_id. Latest, hence -1
            distance = self.data[tracker_id][-1] 

            label = f"#{tracker_id}, {self.class_names[class_id]}, {distance:.2f}"
            labels.append(label)
        return labels
            



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