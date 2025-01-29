import serial
import time
PORT = "/dev/serial0"  # Ändra vid behov
PORT1 = 'PORT = "/dev/ttyS0"'
BAUDRATE = 9600

# testa om data fås med:  minicom -b 9600 -o -D /dev/ttyS0

# mer på: https://chatgpt.com/share/679a92b6-8c4c-8001-9a9c-157cef60513d

with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
    while True:
        data = ser.readline().decode("utf-8", errors="ignore").strip()
        if data:
            print(data)
        else:
            print('Nothing found')
        time.sleep(2)
