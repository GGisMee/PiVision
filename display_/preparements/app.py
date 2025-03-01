import random
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np

class Website:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.app.route('/')(self.index)
        self.points = []
        for i in range(5):
            self.points.append({
                "id": i,
                "x": random.randint(50, 450),
                "y": random.randint(50, 450),
                "dx": random.uniform(-5, 5),
                "dy": random.uniform(-5, 5)
            })

    def index(self):
        return render_template("index.html")

    def generate_points(self):
        i = 0
        while True:
            i+=1
            if i % 10 == 0:
                # self.points.pop()

                self.points.append({
                "id": len(self.points),
                "x": random.randint(50, 450),
                "y": random.randint(50, 450),
                "dx": random.uniform(-5, 5),
                "dy": random.uniform(-5, 5)
            })
            for point in self.points:
                point["x"] += np.random.uniform(-45, 45)
                point["y"] += np.random.uniform(-45, 45)
                point["dx"] = random.uniform(-5, 5)
                point["dy"] = random.uniform(-5, 5)
            
            self.socketio.emit("new_points", self.points)
            time.sleep(1)  # Send updates every second

    def handle_connect(self):
        print("Client connected")

    def run(self):
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.start_background_task(self.generate_points)
        self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    website = Website()
    website.run()
