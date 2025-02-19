from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import time
import threading
import random

class RealTimeServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.current_data = {
            "time": "00:00",
            "d_front": 150,
            "num_now": 3,
            "antal_totalt": 10,
            "extra": ""
        }
        self.running = False  # Track whether updates should run
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True

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

    def update_data(self, time_str: str, d_front: float, num_now: int, warning_status: int):
        """ Update data and notify clients """
        self.current_data.update({
            "time": time_str,
            "d_front": d_front,
            "num_now": num_now,
            "status": warning_status,
            "antal_totalt": random.randint(20, 35)
        })
        self.socketio.emit("update", self.current_data)

    def _update_loop(self):
        """ Background loop to update data periodically """
        while True:
            if self.running:  # Only update when running
                self.update_data(time.strftime("%H:%M:%S"), random.randint(100, 200), random.randint(1, 10), random.randint(0, 1))
            time.sleep(1)

    def run(self, debug=True):
        """ Start the Flask-SocketIO server """
        self.thread.start()
        self.socketio.run(self.app, debug=debug)

if __name__ == "__main__":
    server = RealTimeServer()
    server.run()
