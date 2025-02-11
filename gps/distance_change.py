import serial
import time
import math

PORT = "/dev/ttyAMA0"  # Change this if needed
BAUDRATE = 9600

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two coordinates in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def parse_nmea(data):
    """Extract latitude and longitude from NMEA data."""
    if data.startswith("$GNGGA"):  # GPS fix data
        parts = data.split(",")
        if len(parts) > 5 and parts[2] and parts[4]:  # Ensure valid data
            lat = convert_nmea_to_decimal(parts[2], parts[3])
            lon = convert_nmea_to_decimal(parts[4], parts[5])
            return lat, lon
    return None, None

def convert_nmea_to_decimal(coord, direction):
    """Convert NMEA GPS coordinates to decimal degrees."""
    if not coord or not direction:
        return None
    deg = int(float(coord) // 100)
    minutes = float(coord) % 100
    decimal = deg + (minutes / 60)
    if direction in ["S", "W"]:
        decimal = -decimal
    return decimal

with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
    prev_lat, prev_lon = None, None
    while True:
        data = ser.readline().decode("utf-8", errors="ignore").strip()
        if data:
            lat, lon = parse_nmea(data)
            if lat and lon:
                print(f"Current Position: {lat}, {lon}")
                if prev_lat and prev_lon:
                    distance = haversine(prev_lat, prev_lon, lat, lon)
                    print(f"Distance from previous: {distance:.2f} meters")
                prev_lat, prev_lon = lat, lon
        time.sleep(2)
