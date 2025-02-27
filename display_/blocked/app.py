from flask import Flask, render_template
from flask_socketio import SocketIO
import random
import time

class WebServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.points = {i: self.generate_point(i) for i in range(10)}

        @self.app.route("/")
        def index():
            return render_template("index.j2")

        @self.socketio.on('connect')
        def handle_connect():
            self.socketio.start_background_task(target=self.update_points)

    def generate_point(self, point_id):
        return {
            "id": point_id,
            "x": random.randint(50, 450),
            "y": random.randint(50, 450),
            "dx": random.randint(-20, 20),
            "dy": random.randint(-20, 20)
        }

    def update_points(self):
        while True:
            for point_id in self.points:
                self.points[point_id] = self.generate_point(point_id)
            
            self.socketio.emit('update_points', list(self.points.values()))
            time.sleep(1)

    def run(self, debug=False):
        self.socketio.run(self.app, debug=debug, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    server = WebServer()
    server.run(True)