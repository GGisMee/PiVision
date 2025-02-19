from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from camerafunc.cameraRecorder import get_paths, record_video, get_available_space, get_file_size, takePicture
import subprocess
import threading
import os
import signal
import cv2
import base64

def create_app():
    app = Flask(__name__)
    socketio = SocketIO(app)

    global displayInfo, cameraOn, camera_started
    displayInfo = ''
    cameraOn = False
    camera_started = False

    def display(text: str):
        global displayInfo
        displayInfo += f"{text}<br>"
        socketio.emit('update', displayInfo)

    def start_recording():
        global camera_started, cameraOn
        directory, video_path, _ = get_paths()
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
        display('Closing down')
        os.kill(os.getpid(), signal.SIGINT)

    def save_mp4():
        directory, video_path, video_path_mp4 = get_paths(new=False)
        if get_available_space(directory) >= get_file_size(video_path) + 2:
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-c:v", "copy", video_path_mp4])
        else:
            display('Too little storage left')
            return
        display('MP4 added')

    def snap_picture():
        complete_filename, frame = takePicture(dir='output', filename='testpicture', aspect_ratio=(1920, 1080))
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        display(f"Snapshot taken: {complete_filename} <br> <img src='data:image/jpeg;base64,{jpg_as_text}' width='300'/>")

    def clear_output():
        global displayInfo
        displayInfo = ""

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
        elif data == 'clear_output':
            clear_output()

    return app, socketio

app, socketio = create_app()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
