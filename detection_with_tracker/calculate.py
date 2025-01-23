import numpy as np
import matplotlib.pyplot as plt
import supervision as sv

class DistanceEstimater:
    def __init__(self, parameters):
        self.data: dict = {}

        self.DISTANCE_CONST = 1
        self.FOCAL_LENGTH = 248
     
    def add_detection(self, detections: sv.Detections):
        '''Adds the detections to the data dictionary, which is a dictionary keeping track of the distances to each car.'''
        for i,tracker_id in enumerate(detections.tracker_id):
            distance = self._get_distance(detections.xyxy[i])
            if tracker_id not in self.data.keys():
                self.data[tracker_id] = distance
            else:
                self.data[tracker_id].append(distance)

    def _get_distance(self, xyxy):
        '''Gets the distance from a xyxy box'''
        h_pixels = xyxy[3]-xyxy[1]
        Distance = self.DISTANCE_CONST*h_pixels/self.FOCAL_LENGTH
        return Distance

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