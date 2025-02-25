from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import time
from time import strftime
import threading
import random
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class WebServer:
    def __init__(self, main_manager):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.current_data = {
            "time": "00:00",
            "d_front": '-',
            "d_close": '-',
            "num_now": 3,
            "extra": ""
        }
        self.running = False  # Track whether updates should run
        
        # self.thread = threading.Thread(target=self._update_loop)
        # self.thread.daemon = True

        battery_data = {'level':20}
        self.socketio.emit('battery_update', battery_data)

        self.reference_main_manager = main_manager

        self.start_timestamp = None
        self.last_battery_check_timestamp = None

        @self.app.route("/")
        def index():
            return render_template("index.j2", data=self.get_data())

        @self.app.route("/toggle", methods=["POST"])
        def toggle():
            """ Start/Stop the update loop """
            data = request.get_json()
            if data["action"] == "start":
                if not self.reference_main_manager == None:
                    self.reference_main_manager.start_process()  # Calls MainManager's method
                else:
                    self.log('MainManager not set',1)
                self.log("System started")
                
            else:
                if not self.reference_main_manager == None:
                    self.reference_main_manager.stop_process()  # Calls MainManager's method
                else:
                    self.log('MainManager not set',1)
                print("System stopped")
                self.log("System stopped")

            return jsonify(status="success", running=self.running)

    def log(self, msg:str,type:int=0):
        """ Log a message to the console

        type = index for:
            0 = info
            1 = warning
            2 = error
            3 = critical
            4 = debug"""
        type = ['INFO','WARNING','ERROR','CRITICAL','DEBUG'][type]
        if self.start_timestamp:
            self.start_timestamp - time.time()
            time_str = time.strftime("%H:%M:%S", time.localtime(self.start_timestamp))
        else:
            time_str = strftime("%H:%M:%S")
        self.socketio.emit('log', f'[{type}] {msg} ({time_str})')

    def get_data(self):
        """ Return the latest data """
        return self.current_data

    def update_data(self, d_front: float,d_close:int, num_now: int, warning_status: int):
        """ Update data and notify clients """
        
        # gets time and formats it to either 01:10, or 01:02:30 depending on if hours are necessary
        elapsed_seconds = time.time() - self.start_timestamp
        time_str = time.strftime(
            "%H:%M:%S" if elapsed_seconds >= 3600 else "%M:%S", 
            time.gmtime(elapsed_seconds)
        )
        
        self.current_data.update({
            "time": time_str,
            "d_front": d_front,
            "num_now": num_now,
            "status": warning_status,
            "d_close": d_close
        })
        self.socketio.emit("update", self.current_data)

    def set_start(self):
        """Sets all the functionality to start after startbutton was pressed."""
        self.start_timestamp = time.time()
        self.last_battery_check_timestamp = time.time()        

    def set_stop(self):
        """Sets all the functionality to stop after stopbutton was pressed."""
        self.start_timestamp = None
        self.last_battery_check_timestamp = None

    def run(self, debug=False):
        """ Start the Flask-SocketIO server """
        self.socketio.run(self.app, debug=debug, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    server = WebServer(None)
    server.run(True)