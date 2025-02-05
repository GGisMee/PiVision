from picamera2 import Picamera2, Preview
import time
import cv2
import numpy as np


def takePicture(dir:str = 'output', filename:str='picture', aspect_ratio: tuple[int]= (1920, 1080)):
    camera = Picamera2()

    camera_config = camera.create_still_configuration(main={"size": aspect_ratio, 'format': 'RGB888'})
    camera.configure(camera_config)
    camera.start()
    
    frame = camera.capture_array()

    complete_filename = f'{dir}/{filename}.png'

    cv2.imwrite(filename=complete_filename, img=frame)


if __name__ == '__main__':
    takePicture()

