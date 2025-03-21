import time
import busio
import board
from adafruit_ads1x15.ads1115 import ADS1115

# Initiera I2C
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)

# Läs från AIN0
voltage = ads.read(0)  # Läs från kanal 0
print(f"Spänning på AIN0: {voltage} enheter")
