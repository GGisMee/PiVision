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
import subprocess
import glob
import re


# Variable to stop the recording
recording: bool = False

def get_paths(no_usb = False, new:bool = True):
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
    pattern = f'{directory}/recorded_video*.h264'
    prev_videos: list[str] = glob.glob(pattern)

    # gets the next suffix from the videos in the folder
    nextSuffix = 0
    for video in prev_videos:
        match = re.match(r'.*/recorded_video(\d+)\.h264', video)
        if match:
            currentSuffix = int(match.group(1))
            if new:
                nextSuffix = currentSuffix+1 if nextSuffix <= currentSuffix else nextSuffix
            else:
                nextSuffix = currentSuffix if nextSuffix < currentSuffix else nextSuffix


    video_file_path = os.path.join(directory, f'recorded_video{nextSuffix}.h264')
    video_file_path_mp4 = os.path.join(directory, f'recorded_video{nextSuffix}.mp4')
    return directory, video_file_path, video_file_path_mp4

# Function to check available space in GB
def get_available_space(path):
    total, used, free = shutil.disk_usage(path)
    return free // (2**30)  # Convert from bytes to GB

def get_file_size(file_path:str):
    file_size = os.path.getsize(file_path)  # Size in bytes
    return file_size // (2**30)

# Function to check if there is enough space for recording
def check_space_needed(file_size_in_gb):
    available_space = get_available_space(directory)
    print(f"Available space: {available_space} GB")
    if available_space < file_size_in_gb:
        print("Not enough space to record the video!")
        return False
    return True

# Function to record a video
def record_video(output_file:str, output_dir:str, stopGB: int = 1, camera_on_checker = None, camera_started = None) -> int:
    '''
    
    args:
        stopGB: int = stops recording when stopGB gigabytes are remaining.'''
    global camera
    if not camera_started():
        # Initiera kameran
        camera = Picamera2()

        # Ställ in för att visa förhandsvisningen
        camera.start_preview(Preview.NULL)

    # Konfigurera kameran för video
    camera.configure(camera.create_video_configuration())

    # Starta kameran
    camera.start()

    # Skapa encoder för H264-video
    encoder = H264Encoder()

    # Tillfällig H.264-utdatafil
    output = FileOutput(output_file)

    # Starta inspelningen med encoder och utdata
    camera.start_recording(encoder, output)

    # Filma i 10 sekunder
    print("Recording started")
    while get_available_space(output_dir) > stopGB:
        if camera_on_checker and not camera_on_checker():
            print('stopped by user')
            break
        sleep(1)
    print("Recording stopped")

    # Stoppa inspelningen
    camera.stop_recording()

    return 0

def convert():
    # Konvertera H.264 till MP4 med ffmpeg
    mp4_output_file = "video_test2.mp4"
    h264_output_file = ''
    subprocess.run([
    "ffmpeg", "-y", "-i", h264_output_file, "-c:v", "copy", mp4_output_file
    ])

    print(f"Video saved as {mp4_output_file}")


if __name__ == "__main__":
    directory, video_path, video_file_path_mp4 = get_paths(False, True)
    print(f'Video Path: {video_path}')
    free_space = get_available_space(directory)
    print(f'Free space: {free_space} GB')
    h264_output_file = record_video(video_path,directory, 3, lambda: True, lambda: False)
