import time
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from numpy import mean
from collections import deque

class VoltageTester:
    def __init__(self):
        # Försök att ansluta till ADS1015/ADS1115
        self.i2c = board.I2C()  
        try:
            self.ads = ADS.ADS1015(self.i2c)
            self.ads.gain = 1
            self.chan = AnalogIn(self.ads, ADS.P0)
        except ValueError as e:
            print(f"I2C Error: {e}")
            self.ads = None
            self.chan = None

        # Sensorparametrar
        self.VOLTAGE_DIVIDER_RATIO = 5  
        self.CELLS = 3
        self.values_len = 4
        self.values = deque(maxlen=self.values_len)

        # Spänningsgränser (LiPo-batteri)
        self.max_voltage = 4.2
        self.min_voltage = 3.8

        self.test_regularity = 5  # sekunder mellan tester
        self.latest_voltagetest_timestamp = 0
        self.voltage_percentage = 0  # Initiera till 0%

    def _read_battery_voltage(self):
        """Läser aktuell batterispänning via ADS1015."""
        if not self.chan:
            print("⚠️  Ingen ADS1015 hittades! Kontrollera kopplingar.")
            return None
        try:
            sensor_voltage = self.chan.voltage
            return (sensor_voltage * self.VOLTAGE_DIVIDER_RATIO) / self.CELLS
        except Exception as e:
            print(f"⚠️  Fel vid läsning av spänning: {e}")
            return None

    def get_current_voltage(self):
        """Returnerar ett medelvärde av de senaste mätningarna."""
        new_voltage = self._read_battery_voltage()
        if new_voltage is None:
            return None
        self.values.append(new_voltage)
        return mean(self.values) if self.values else None

    def get_percentage_left(self):
        """Beräknar hur mycket batteri som finns kvar i procent."""
        if not self.check_if_time():
            return self.voltage_percentage

        current_voltage = self.get_current_voltage()
        if current_voltage is None:
            return self.voltage_percentage  # Behåll senaste värdet

        percentage = int(round(100 * (current_voltage - self.min_voltage) / (self.max_voltage - self.min_voltage)))
        self.voltage_percentage = max(0, min(100, percentage))  # Begränsa mellan 0-100%
        return self.voltage_percentage

    def check_if_time(self):
        """Kontrollerar om det är dags för en ny mätning."""
        if time.time() - self.latest_voltagetest_timestamp > self.test_regularity:
            self.latest_voltagetest_timestamp = time.time()
            return True
        return False


if __name__ == '__main__':
    vtest = VoltageTester()
    try:
        while True:
            bat_percentage = vtest.get_percentage_left()
            print(f"🔋 Batterinivå: {bat_percentage}%")
            time.sleep(1)
    except KeyboardInterrupt:
        print("🚪 Avslutar programmet.")
