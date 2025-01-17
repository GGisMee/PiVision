# https://chatgpt.com/share/678aae5b-3068-8001-899c-0ebdb2f98096

import time
import board
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn



# Initialize the ADS1115 ADC
i2c = board.I2C()  # Use default I2C pins (SDA: GPIO2, SCL: GPIO3)
ads = ADS.ADS1015(i2c)

# Configure the ADS1115
ads.gain = 1

# Connect the sensor to A0
chan = AnalogIn(ads, ADS.P0)

# Voltage sensor parameters
VOLTAGE_DIVIDER_RATIO = 5  # Voltage sensor divides input voltage by 5

def read_battery_voltage():
    # Read the voltage from the sensor
    sensor_voltage = chan.voltage
    # Scale it back to the actual input voltage
    battery_voltage = sensor_voltage * VOLTAGE_DIVIDER_RATIO
    return battery_voltage

# Main loop
try:
    while True:
        battery_voltage = read_battery_voltage()
        print(f"Battery Voltage: {battery_voltage/3:.2f} V")
        time.sleep(1)  # Wait 1 second before the next reading

except KeyboardInterrupt:
    print("Exiting program.")
