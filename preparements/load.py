import cv2
import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np

class ImageGrabber:
    def __init__(self, videoPath:str):
        self.videoPath = videoPath
        self.cap = cv2.VideoCapture(self.videoPath)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    
    def getVideoPaths():
        '''Gets the video paths inside videoSource'''
        current_file_dir = os.getcwd()
        # parent_dir = os.path.dirname(current_file_dir)
        video_source_path = os.path.join(current_file_dir, 'sources/videoSource')
        videoPaths = glob.glob(f'{video_source_path}/*')
        return videoPaths
    
    def getImg(self, frame_index:str = 'Next') -> np.ndarray:
        '''Gets an image from the video
        
        Variables:
            frame_index: index:int or str "Random" or str "Next"
            
        Returns:
            frame: np.ndarray'''

        if frame_index.isdigit():
            if not 0 <= frame_index <= self.total_frames:
                raise ValueError('Frame does not exist, since it is out of bound')
        if frame_index == 'Random':
            frame_index = random.randint(0, self.total_frames)
            

        # test if the video is found.
        if not self.cap.isOpened():
            return 1
        
        if frame_index != 'Next':
            # sets the position for the frame to be grabbed.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # reads the frame
        ret, frame = self.cap.read() # here ret is bool if it worked.

        if not ret:
            return 1

        # resizing the frame
        yLen, xLen, _ = frame.shape
        Len = min(xLen, yLen)
        frame = frame[int((yLen-Len)/2):int((yLen-Len)/2+Len), int((xLen-Len)/2):int((xLen-Len)/2+Len)]
        frame = cv2.resize(frame, (640, 640))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        

        self.frame = frame
        return frame
    def viewFrame(self, frame:np.ndarray=None):
        '''Views the frame which was grabbed or the one inputed'''
        if frame is None:
            plt.imshow(self.frame)
        else:
            plt.imshow(self.frame)
        plt.show()
        

if __name__ == '__main__':
    path = ImageGrabber.getVideoPaths()[0]
    grabber = ImageGrabber(path)
    grabber.getImg('Random')
    grabber.viewFrame()
    plt.show()
    