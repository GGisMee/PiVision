import numpy as np
import matplotlib.pyplot as plt


def derive(y):
    return np.insert((y[1:]-y[:-1]),0,0)
    

W_actual:float = 1.8
focal:int=248
def viewData(distance:np.ndarray, speed: np.ndarray, acceleration:np.ndarray):
    plt.yticks(np.arange(-1,7,0.2))
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.plot(distance, color = 'red')
    plt.plot(speed, color = 'blue')
    plt.plot(acceleration, color = 'green')
    plt.show()

def getDistance(xyxys:dict):
    distData = {}
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
    # get data:
    xyxy = {'1': [np.array([     229.51,      342.44,      334.56,       422.7], dtype=np.float32), np.array([     230.51,      341.45,      332.74,      420.77], dtype=np.float32), np.array([     228.49,      341.93,      332.37,      421.59], dtype=np.float32), np.array([     231.81,      344.12,      332.29,      422.57], dtype=np.float32), np.array([     230.17,      341.03,  
        330.02,      419.38], dtype=np.float32), np.array([      230.1,      338.63,      330.87,      416.03], dtype=np.float32), np.array([     230.09,      343.72,       330.2,      419.02], dtype=np.float32), np.array([     234.25,      345.26,    
      330.23,      420.88], dtype=np.float32)], '2': [np.array([     491.97,      332.16,       575.4,      397.16], dtype=np.float32)], '3': [np.array([     467.15,      327.12,      623.31,      402.18], dtype=np.float32)]}
    distances:dict = getDistance(xyxy)
    speed, acceleration = get_speed_and_acceleration(distances)
    viewData(distances['1'], speed['1'], acceleration['1'])