from flask import Flask, render_template
from flask_socketio import SocketIO
import random
import time

class WebServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        @self.app.route("/")
        def index():
            return render_template("index.j2")
        
        @self.socketio.on('connect')
        def handle_connect():
            self.socketio.start_background_task(target=self.generate_points)
    
    def generate_points(self):
        while True:
            points = [
                {
                    "x": random.randint(50, 450), 
                    "y": random.randint(50, 450),
                    "dx": random.randint(-20, 20), 
                    "dy": random.randint(-20, 20)
                }
                for _ in range(10)
            ]
            self.socketio.emit('new_points', points)
            time.sleep(1)
    
    def run(self, debug=False):
        """ Start the Flask-SocketIO server """
        self.socketio.run(self.app, debug=debug, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    server = WebServer()
    server.run(True)
