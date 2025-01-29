import serial
import pynmea2

# Ange rätt seriell port (för Raspberry Pi är det oftast /dev/serial0 eller /dev/ttyS0)
PORT = "/dev/serial0"
BAUDRATE = 9600  # Standardhastighet för BN-220

def read_gps():
    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("$GNGGA") or line.startswith("$GPGGA"):  # GPS-fixdata
                try:
                    msg = pynmea2.parse(line)
                    latitude = msg.latitude
                    longitude = msg.longitude
                    print(f"Lat: {latitude}, Lon: {longitude}")
                except pynmea2.ParseError:
                    pass

if __name__ == "__main__":
    read_gps()
