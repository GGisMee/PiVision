from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import threading
import random

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Shared data that updates in real-time
current_data = {
    "time": "00:00",
    "d_front": 150,
    "num_now": 3,
    "antal_totalt": 10,
    "extra": ""
}

def get_data():
    """ Return the latest data """
    return current_data

def update_data(time:str, d_front:float, num_now:int, warning_status: int):
    """ Simulate real-time updates """

    current_data["time"] = time
    current_data["d_front"] = d_front  # Simulate sensor data
    current_data["num_now"] = num_now
    current_data["status"] = warning_status
    current_data["antal_totalt"] = random.randint(20, 35)
        
    # Send updated data to all connected clients
    socketio.emit('update', current_data)
    time.sleep(1)  # Update every second

@app.route("/")
def index():
    return render_template("index.html", data=get_data())

if __name__ == "__main__":
    # Start the update function in a background thread
    thread = threading.Thread(target=update_data)
    thread.daemon = True
    thread.start()

    socketio.run(app, debug=True)
