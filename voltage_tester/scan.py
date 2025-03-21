import board
import busio

i2c = busio.I2C(board.SCL, board.SDA)

while not i2c.try_lock():
    pass

devices = i2c.scan()
print("I2C devices found:", [hex(device) for device in devices])
i2c.unlock()
