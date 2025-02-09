from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import random
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def send_updates():
    """ Function to send updates continuously """
    while True:
        value = random.randint(1, 100)  # Replace with real-time value
        socketio.emit('update', {'value': value})
        time.sleep(1)  # Update every second

@app.route('/')
def index():
    return """
    <html>
        <body>
            <h1>Live Data: <span id="data">Waiting...</span></h1>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <script>
                var socket = io();
                socket.on('update', function(data) {
                    document.getElementById("data").innerText = data.value;
                });
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":
    thread = threading.Thread(target=send_updates)
    thread.daemon = True
    thread.start()
    socketio.run(app, debug=True)
