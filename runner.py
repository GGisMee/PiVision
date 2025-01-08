# https://chatgpt.com/share/677ed560-6fa4-8001-9588-3bd453b0bf3d


import subprocess
from picamera2 import Picamera2
import time

def check_internet():
    def get_response():
        response = subprocess.run(['ping', '-c', '1', '8.8.8.8'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return response.returncode

    try:
        # Attempt to ping Google's DNS server
        while get_response() != 0:
            time.sleep(3)
    except subprocess.CalledProcessError:
        print("Error in checking internet connection.")
        return 1
    print('Internet available')
        


def check_camera():
    def try_cam():
        try:
            cam = Picamera2()
            cam.start_preview()
            return 0
        except Exception as e:
            return 1
    while try_cam():
        time.sleep(3)
    print('Camera available')
        

check_internet()
check_camera()

# Command to run the file using python -m
command = ['python', '-m', 'web.app']  # Assuming 'flask.app' is the module you want to run
subprocess.run(command)
