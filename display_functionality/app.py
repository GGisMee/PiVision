from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import time
import logging
import numpy as np
import random
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_area_in_pixels(area_p: list[int], forward_length_m: int):
    ratio_p_div_m = area_p[1] / forward_length_m
    ID_to_area_p = {
        2: (np.array([1.75, 4.55]) * ratio_p_div_m).round(),
        5: (np.array([2.55, 12]) * ratio_p_div_m).round(),
        7: (np.array([2.4, 16]) * ratio_p_div_m).round()
    }
    return ID_to_area_p, ratio_p_div_m

class WebServer:
    def __init__(self, main_manager):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.current_data = {
            "time": "00:00",
            "d_front": '-',
            "d_close": '-',
            "num_now": '-',
            'status': 0
        }
        self.running = False
        self.reference_main_manager = main_manager
        self.start_timestamp = None
        self.last_battery_check_timestamp = None

        self.FORWARD_LEN = 20 # the length forward which will be displayed on the website

        self.ratio_p_div_m:float = None

        self.ID_to_color = {1:[122,122,122], 2:[14,160,2], 4:[255,0,3]}

        self.points = []
        for i in range(5):
            self.points.append({
                "id": i,
                "x": random.randint(50, 450),
                "y": random.randint(50, 450),
                "dx": random.uniform(-5, 5),
                "dy": random.uniform(-5, 5)
            })


        @self.app.route("/")
        def index():
            return render_template("index.j2", data=self.current_data)

        @self.socketio.on('toggle')
        def handle_toggle(data):
            action = data.get("action")
            if action == "start":
                if self.reference_main_manager:
                    self.reference_main_manager.start_process()
                self.running = True
                self.set_start()
                self.log("System started")
            else:
                if self.reference_main_manager:
                    self.reference_main_manager.stop_process()
                self.running = False
                self.set_stop()
                self.log("System stopped")
            self.socketio.emit('status_update', {"running": self.running})

        @self.socketio.on('div_area')
        def handle_div_area(data):
            width = data['width']
            height = data['height']
            self.ID_to_area_p, self.ratio_p_div_m = get_area_in_pixels((width,height), self.FORWARD_LEN)

    # canvas functionality
    def send_points(self, latest_data:np.ndarray, ID_to_color):
            '''
            Input:
                latest_data: [[id, dx[-1], dy[-1], vx,vy]]'''

            for car in latest_data:
                self.points.append({
                "id": car[0],
                "x": car[1],
                "y": car[2],
                "dx": car[3],
                "dy": car[4]
                })
            
            self.socketio.emit("new_points", json.dumps(self.points))
            time.sleep(1)  # Send updates every second

    def handle_connect(self):
        print("Client connected")


    # other
    def log(self, msg: str, type: int = 0):
        log_types = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG']
        log_type = log_types[min(type, 4)]
        time_str = time.strftime("%H:%M:%S")
        self.socketio.emit('log', f'[{log_type}] {msg} ({time_str})')

    def process_data(self, latest_data:np.ndarray):
        '''Process the data to prepare it to be displayed as pixels
        
        Input:
            latest_data = [tracker_id, dx, dy, vx, vy]'''
        actual_data_m = latest_data[:,1:]*self.ratio_p_div_m
        ids = latest_data[:,0]
        processed_data = np.column_stack((ids, actual_data_m))
        return processed_data


    def update_data(self, d_front: float, d_close: int, num_now: int,latest_data:np.ndarray,ID_to_color:dict, warning_status: int):
        elapsed_seconds = time.time() - self.start_timestamp if self.start_timestamp else 0
        time_str = time.strftime("%H:%M:%S" if elapsed_seconds >= 3600 else "%M:%S", time.gmtime(elapsed_seconds))
        self.ID_to_color = ID_to_color
        processed_data = self.process_data(latest_data)
        self.current_data.update({
            "time": time_str,
            "d_front": d_front,
            "num_now": num_now,
            "status": warning_status,
            "latest_data": processed_data,
            "d_close": d_close,
        })
        self.send_points(latest_data, ID_to_color)
        self.socketio.emit("update", self.current_data)

    def set_start(self):
        self.start_timestamp = time.time()
        self.last_battery_check_timestamp = time.time()

    def set_stop(self):
        self.start_timestamp = None
        self.last_battery_check_timestamp = None

    def run(self, debug=False):
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.run(self.app, debug=debug, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    server = WebServer(None)
    server.run(True)
