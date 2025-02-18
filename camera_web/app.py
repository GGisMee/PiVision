from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from camerafunc.cameraRecorder import get_paths, record_video, get_available_space, get_file_size, takePicture
import subprocess
import threading
import os
import signal

app = Flask(__name__)
socketio = SocketIO(app)

global displayInfo, cameraOn
displayInfo = ''
cameraOn = False
camera_started = False



def display(text:str):
    global displayInfo
    # adds the text
    displayInfo += f"{text}<br>"
    # Emit the updated displayInfo to the client
    socketio.emit('update', displayInfo)

# Define the function for each button
def start_recording():
    global camera_started
    global cameraOn
    directory, video_paths = get_paths()
    video_path = video_paths['.h264']

    display(f'path: {video_path}')
    cameraOn = True
    display('Recording started')
    recording_thread = threading.Thread(target=record_video, args=(video_path, directory, 3, lambda: cameraOn, lambda: camera_started))    
    recording_thread.start()


def stop_recording():
    global cameraOn
    cameraOn = False
    display('Recording stopped')

def stop_button():

    display('closing down')
    os.kill(os.getpid(), signal.SIGINT)


def save_mp4():
    directory, video_paths = get_paths(new=False)
    video_path = video_paths['.h264']
    video_path_mp4 = video_paths['.mp4']
    if get_available_space(directory) >= get_file_size(video_path)+2:
        subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-c:v", "copy", video_path_mp4
        ])
    else:
        display('Too little storage left')
        return
    display('mp4 added')

def snap_picture():
    complete_filename, frame = takePicture(dir='output', filename='testpicture', aspect_ratio=(1920,1080))
    

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('button_pressed')
def handle_button_pressed(data):
    if data == 'start_recording':
        start_recording()
    elif data == 'stop_recording':
        stop_recording()
    elif data == 'save_mp4':
        save_mp4()
    elif data == 'stop_button':
        stop_button()
    elif data == 'snap_picture':
        snap_picture()


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)