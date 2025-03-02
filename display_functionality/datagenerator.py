import numpy as np
import time
import threading
import random

class DataGenerator:
    def __init__(self, webserver, num_vehicles=5, update_interval=1.0):
        self.webserver = webserver
        self.num_vehicles = num_vehicles
        self.update_interval = update_interval
        self.running = False

        # Initialize vehicle data: [id, x, y, dx, dy]
        self.vehicles = np.array([
            [i, random.uniform(0, 40), random.uniform(0, 40), 0, 0]
            for i in range(num_vehicles)
        ])
    
    def start(self):
        """Start the data simulation in a separate thread."""
        self.running = True
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop the data simulation."""
        self.running = False
    
    def update_vehicles(self):
        """Increment small changes to dx, dy and update vehicle positions."""
        for vehicle in self.vehicles:
            vehicle[3] += random.uniform(-1, 1)  # Small dx change
            vehicle[4] += random.uniform(-1, 1)  # Small dy change
            vehicle[1] += vehicle[3]  # Update x
            vehicle[2] += vehicle[4]  # Update y
        
        return self.vehicles
    
    def run(self):
        """Continuously generate and send test data."""
        while self.running:
            latest_data = self.update_vehicles()
            ID_to_color = {i: [random.randint(0, 255) for _ in range(3)] for i in range(self.num_vehicles)}
            self.webserver.update_data(
                d_front=random.uniform(0, 50),
                d_close=random.randint(0, 20),
                num_now=self.num_vehicles,
                latest_data=latest_data,
                ID_to_color=ID_to_color,
                warning_status=random.randint(0, 9)
            )
            time.sleep(self.update_interval)

if __name__ == "__main__":
    from display_functionality.app import WebServer  # Assuming your server file is named web_server.py
    
    server = WebServer(None)
    generator = DataGenerator(server, num_vehicles=5, update_interval=1)
    generator.start()
    server.run(debug=True)
