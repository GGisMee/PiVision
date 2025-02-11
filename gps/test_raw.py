import serial
import time

PORT = "/dev/ttyAMA0"  # Change this if needed
BAUDRATE = 9600

# Check if data is being received: 
# cat /dev/ttyAMA0 | grep GSV (to check for satellite data)

# Test with minicom:
# cd /; minicom -b 9600 -o -D /dev/ttyAMA0

# cgps -s 

# For more info, see: https://chatgpt.com/share/679a92b6-8c4c-8001-9a9c-157cef60513d


def parse_nmea(data):
    """Function to parse NMEA data and extract useful information."""
    if data.startswith("$GNGGA"):  # GPS fix data
        print("GNGGA - GPS Fix Data: ", data)
    elif data.startswith("$GNRMC"):  # Recommended Minimum Specific GPS/Transit data
        print("GNRMC - GPS Data: ", data)
    elif data.startswith("$GPGSV"):  # Satellites in view
        print("GPGSV - Satellite Information: ", data)
    elif data.startswith("$GNVTG"):  # Course over ground and speed data
        print("GNVTG - Speed and Course Data: ", data)
    elif data.startswith("$GNGLL"):  # Geographic position data
        print("GNGLL - Geographic Position: ", data)
    else:
        print("Other NMEA Data: ", data)

with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
    while True:
        data = ser.readline().decode("utf-8", errors="ignore").strip()
        if data:
            parse_nmea(data)
        else:
            print('Nothing found')
        time.sleep(2)
