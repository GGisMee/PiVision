from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import threading
import random


class WebManager:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Registrera route
        self.app.route("/")(self.index)

        # Shared data that updates in real-time
        self.current_data = {
            "time": "00:00",
            "d_front": 150,
            "num_now": 3,
            "status": 0,
            "antal_totalt": 10,
            "d_near": 150,
        }

    def get_data(self):
        """ Return the latest data """
        return self.current_data

    def update_data(self, time:str='00:00', d_front:float='-', num_now:int='-', warning_status: int=0, d_near:int='-'):
        """ Simulate real-time updates """

        self.current_data["time"] = time
        self.current_data["d_front"] = d_front
        self.current_data["num_now"] = num_now
        self.current_data["status"] = warning_status
        self.current_data["d_near"] = d_near
        # Skicka uppdaterade data till alla klienter
        self.socketio.emit('update', self.current_data)
        time.sleep(1)  # Uppdatera varje sekund

    def index(self):
        return render_template("index.html", data=self.get_data())


if __name__ == "__main__":
    webmanager = WebManager()

    # Starta update_data() i en bakgrundstråd
    thread = threading.Thread(target=webmanager.update_data, daemon=True)
    thread.start()

    webmanager.socketio.run(webmanager.app, debug=True)
