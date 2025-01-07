import os
import shutil
import cv2
import time
from datetime import datetime
import threading
from time import sleep
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from time import sleep
import subprocess
import glob
import re


def get_paths(no_usb = False):
    '''Gets both the path for the new video and for the directory for the video'''
    # Define the path to your external USB
    usb_path = '/media/gustavgamstedt/Samsung USB'  # Change this to the correct path of your USB

    # Checks if the usb is plugged in.
    if not os.path.isdir(usb_path) or no_usb:
        print('No USB')
        directory = 'resources/videos'
    else:
        print('USB')
        directory = usb_path

    # gets all previous videopaths.
    pattern = f'{directory}/recording*.h264'
    prev_videos: list[str] = glob.glob(pattern)

    # gets the next suffix from the videos in the folder
    nextSuffix = 0
    for video in prev_videos:
        match = re.match(r'.*/recording(\d+)\.h264', video)
        if match:
            currentSuffix = int(match.group(1))
            nextSuffix = currentSuffix+1 if nextSuffix < currentSuffix else nextSuffix

    video_file_path = os.path.join(directory, f'recorded_video{nextSuffix}.h264')
    return directory, video_file_path

# Function to check available space in GB
def get_available_space(path):
    total, used, free = shutil.disk_usage(path)
    return free // (2**30)  # Convert from bytes to GB

# Function to check if there is enough space for recording
def check_space_needed(file_size_in_gb):
    available_space = get_available_space(directory)
    print(f"Available space: {available_space} GB")
    if available_space < file_size_in_gb:
        print("Not enough space to record the video!")
        return False
    return True

# Function to record a video
def record_video():


    # Initiera kameran
    camera = Picamera2()

    # Ställ in för att visa förhandsvisningen
    camera.start_preview(Preview.QT)

    # Konfigurera kameran för video
    camera.configure(camera.create_video_configuration())

    # Starta kameran
    camera.start()

    # Skapa encoder för H264-video
    encoder = H264Encoder()

    # Tillfällig H.264-utdatafil
    output = FileOutput(h264_output_file)

    # Starta inspelningen med encoder och utdata
    camera.start_recording(encoder, output)

    # Filma i 10 sekunder
    print("Recording started")
    w
    print("Recording stopped")

    # Stoppa inspelningen
    camera.stop_recording()

    return h264_output_file

def convert():
    # Konvertera H.264 till MP4 med ffmpeg
    mp4_output_file = "video_test2.mp4"
    h264_output_file = ''
    subprocess.run([
    "ffmpeg", "-y", "-i", h264_output_file, "-c:v", "copy", mp4_output_file
    ])

    print(f"Video saved as {mp4_output_file}")

if __name__ == "__main__":
    directory, video_path = get_paths(False)
    print(f'Video Path: {video_path}')
    # free_space = get_available_space(directory)
    # print(f'Free space: {free_space} GB')
    # h264_output_file = record_video()
