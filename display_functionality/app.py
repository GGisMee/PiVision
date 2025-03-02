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

        self.ratio_p_div_m = 1 #! fix later
        self.ID_to_color = {}

        self.current_vehicle_ids = set()

        self.width = None
        self.height = None
        
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
            self.width = data['width']
            self.height = data['height']
            self.ID_to_area_p, self.ratio_p_div_m = get_area_in_pixels((self.width,self.height), self.FORWARD_LEN)
            self.log(f"Canvas dimensions set: {self.width}x{self.height} pixels")

    def send_vehicle_data(self, processed_data):
        """
        Send vehicle data to the client.

        Args:
            processed_data: Processed data array with [id, x, y, dx, dy] for each vehicle
        """
        if processed_data.size == 0:
            # No vehicles to display, clear all
            self.current_vehicle_ids = set()
            self.socketio.emit("vehicle_update", {"vehicles": []})
            return

        vehicles = []
        for vehicle in processed_data:
            vehicle_id = int(vehicle[0])
            color = self.ID_to_color.get(vehicle_id, "#CCCCCC")  # Default to gray if no color

            vehicles.append({
                "id": vehicle_id,
                "x": float(vehicle[1])+self.width/2, # put it in the middle if 0
                "y": float(vehicle[2]),
                "dx": float(vehicle[3]),
                "dy": float(vehicle[4]),
                "color": color if isinstance(color, str) else f"rgb({color[0]}, {color[1]}, {color[2]})"
            })
        self.socketio.emit("vehicle_update", {"vehicles": vehicles})

    def handle_connect(self):
        self.log("Client connected")

    def log(self, msg: str, type: int = 0):
        log_types = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG']
        log_type = log_types[min(type, 4)]
        time_str = time.strftime("%H:%M:%S")
        self.socketio.emit('log', f'[{log_type}] {msg} ({time_str})')

    def process_data(self, latest_data:np.ndarray):
        '''Process the data to prepare it to be displayed as pixels
        
        Input:
            latest_data = [tracker_id, dx, dy, vx, vy]
            
        Returns:
            processed_data = [tracker_id, x_pixel, y_pixel, dx_norm, dy_norm]
        '''
        if latest_data.size == 0:
            return np.array([])
            
        # Apply ratio to convert meters to pixels
        positions = latest_data[:, 1:3] * self.ratio_p_div_m
        
        # Normalize velocity vectors for direction indicators
        velocities = latest_data[:, 3:5]
        velocity_magnitudes = np.linalg.norm(velocities, axis=1, keepdims=True)
        # Avoid division by zero
        normalized_velocities = np.where(
            velocity_magnitudes > 0.001,
            velocities / velocity_magnitudes,
            np.zeros_like(velocities)
        )
        
        # Combine IDs with processed data
        ids = latest_data[:, 0].reshape(-1, 1)
        processed_data = np.hstack((ids, positions, normalized_velocities))
        
        return processed_data

    def update_data(self, d_front: float, d_close: int, num_now: int, latest_data: np.ndarray, ID_to_color: dict, warning_status: int):
        """
        Update dashboard data and send to clients.
        
        Args:
            d_front: Distance to front vehicle in meters
            d_close: Distance to closest vehicle in meters
            num_now: Number of vehicles detected
            latest_data: Raw vehicle data array [id, dx, dy, vx, vy]
            ID_to_color: Dictionary mapping vehicle IDs to display colors
            warning_status: Warning level (0-9)
        """
        # Calculate elapsed time
        elapsed_seconds = time.time() - self.start_timestamp if self.start_timestamp else 0
        time_str = time.strftime("%H:%M:%S" if elapsed_seconds >= 3600 else "%M:%S", time.gmtime(elapsed_seconds))
        
        # Save color dictionary
        self.ID_to_color = ID_to_color
        
        # Process vehicle data for rendering
        processed_data = self.process_data(latest_data)
        
        # Update current data for dashboard
        self.current_data.update({
            "time": time_str,
            "d_front": d_front,
            "num_now": num_now,
            "status": warning_status,
            "d_close": d_close,
        })
        
        # Send vehicle data for canvas rendering
        self.send_vehicle_data(processed_data)
        
        # Send dashboard updates
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