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

       
    def run(self, debug=False):
        self.socketio.run(self.app, debug=debug, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    server = WebServer()
    server.run(True)