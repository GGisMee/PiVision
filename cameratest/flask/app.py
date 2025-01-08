from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from cameraRecorder import get_paths, record_video, get_available_space, get_file_size
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
def function_one():
    global camera_started
    global cameraOn
    directory, video_path, _ = get_paths(False)
    display(f'path: {video_path}')
    cameraOn = True
    display('Recording started')
    recording_thread = threading.Thread(target=record_video, args=(video_path, directory, 3, lambda: cameraOn, lambda: camera_started))    
    recording_thread.start()
    camera_started = True


def function_two():
    global cameraOn
    cameraOn = False
    display('Recording stopped')

def stop_button():

    display('closing down')
    os.kill(os.getpid(), signal.SIGINT)


def function_three():
    directory, video_path, video_path_mp4 = get_paths(new=False)
    if get_available_space(directory) >= get_file_size(video_path)+2:
        subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-c:v", "copy", video_path_mp4
        ])
    else:
        display('Too little storage left')
        return
    display('mp4 added')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('button_pressed')
def handle_button_pressed(data):
    if data == 'button_one':
        function_one()
    elif data == 'button_two':
        function_two()
    elif data == 'button_three':
        function_three()
    elif data == 'stop_button':
        stop_button()

def run():
    socketio.run(app, debug=True)
    

if __name__ == '__main__':
    socketio.run(app, debug=True)
