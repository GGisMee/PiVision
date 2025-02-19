# https://chatgpt.com/share/677ed560-6fa4-8001-9588-3bd453b0bf3d

from camera_web.app import create_app
import subprocess
from picamera2 import Picamera2
import time
from datetime import datetime
import sys
print(f"Python executable: {sys.executable}")


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
            cam.stop()
            cam.close()
            return 0
        except Exception as e:
            return 1
    while try_cam():
        time.sleep(3)
    print('Camera available')
        

check_internet()
check_camera()

def log_timestamp():
    try:
        with open("/home/gustavgamstedt/Desktop/Programming/PiVision/run.txt", "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"Script ran at: {timestamp}\n")
            print(f"Timestamp logged: {timestamp}")
    except Exception as e:
        print(f"Failed to log timestamp: {e}")


# Command to run the file using python -m
#command = ['python', '-m', 'camera_web.app']  # Assuming 'flask.app' is the module you want to run
#subprocess.Popen(command)
app, socketio = create_app()
socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
log_timestamp()

