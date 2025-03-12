# https://chatgpt.com/share/678aae5b-3068-8001-899c-0ebdb2f98096

import time
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from numpy import mean
from collections import deque

class VoltageTester:
    def __init__(self):
        
        # Initialize the ADS1115 ADC
        self.i2c = board.I2C()  # Use default I2C pins (SDA: GPIO2, SCL: GPIO3)
        try:
            self.ads = ADS.ADS1015(self.i2c)
            self.ads.gain = 1
            self.chan = AnalogIn(self.ads, ADS.P0)


        except ValueError:
            self.ads = None
            self.chan = None


        # Voltage sensor parameters
        self.VOLTAGE_DIVIDER_RATIO = 5  # Voltage sensor divides input voltage by 5
        self.CELLS = 3

        self.values_len = 4
        self.values = deque(maxlen=self.values_len)

        self.max_voltage = 4.2
        self.min_voltage = 3.8

        self.test_regularity = 5 # seconds
        self.latest_voltagetest_timestamp = None
        self.voltage_procentage = self.get_procentage_left()

    def _read_battery_voltage(self):
        ''''''
        # Read the voltage from the sensor
        try:
            sensor_voltage = self.chan.voltage
        except AttributeError:
            return None
        # Scale it back to the actual input voltage
        battery_voltage = sensor_voltage * self.VOLTAGE_DIVIDER_RATIO/self.CELLS
        return battery_voltage

    def get_current_voltage(self):
        '''Gets the latest voltage and returns the mean voltage from the self.values_len last items'''
        new_voltage = self._read_battery_voltage()
        if not new_voltage:
            return None
        self.values.append(new_voltage)
        return mean(self.values)
    
    def get_procentage_left(self):
        if not self.check_if_time():
            return self.voltage_procentage
        current_voltage = self.get_current_voltage()
        if not current_voltage:
            return None
        procentage = int(round(100*(current_voltage-self.min_voltage)/(self.max_voltage-self.min_voltage)))
        if procentage < 0:
            return 0
        if procentage > 100:
            return 100
        return procentage
    
    def check_if_time(self):
        if not self.latest_voltagetest_timestamp:
            self.latest_voltagetest_timestamp = time.time()
            return True
        
        time_now = time.time()
        if time_now-self.test_regularity > self.latest_voltagetest_timestamp:
            self.latest_voltagetest_timestamp = time_now
            return True
        return False




if __name__ == '__main__':
    # Main loop
    vtest = VoltageTester()
    try:
        while True:
            bat_procentage = vtest.get_procentage_left()
            print(f"Battery at {bat_procentage}% left")
            time.sleep(1)  # Wait 1 second before the next reading

    except KeyboardInterrupt:
        print("Exiting program.")
