from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import time
import threading
import random

class WebServer:
    def __init__(self):
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
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True

        battery_data = {'level':20}
        self.socketio.emit('battery_update', battery_data)

        @self.app.route("/")
        def index():
            return render_template("index.html", data=self.get_data())

        @self.app.route("/toggle", methods=["POST"])
        def toggle():
            """ Start/Stop the update loop """
            data = request.get_json()
            if data["action"] == "start":
                self.running = True
                print("System started")
            else:
                self.running = False
                print("System stopped")
            return jsonify(status="success", running=self.running)

    def get_data(self):
        """ Return the latest data """
        return self.current_data

    def update_data(self, time_str: str, d_front: float,d_close:int, num_now: int, warning_status: int):
        """ Update data and notify clients """
        self.current_data.update({
            "time": time_str,
            "d_front": d_front,
            "num_now": num_now,
            "status": warning_status,
            "d_close": d_close
        })
        self.socketio.emit("update", self.current_data)

    def _update_loop(self):
        """ Background loop to update data periodically """
        while True:
            if self.running:  # Only update when running
                self.update_data(time.strftime("%H:%M:%S"), random.randint(100, 200), random.randint(100, 200), random.randint(1, 10), random.randint(0, 1))
            time.sleep(1)

    def set_start(self):
        """Sets all the functionality to start after button was pressed."""

    def run(self, debug=True):
        """ Start the Flask-SocketIO server """
        self.thread.start()
        self.socketio.run(self.app, debug=debug)

if __name__ == "__main__":
    server = WebServer()
    server.run()
