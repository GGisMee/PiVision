import lgpio

class BuzzerManager:
    def __init__(self, PIN:int = 3,use_buzzer:bool=False):
        self.use_buzzer = use_buzzer
        self.PIN = PIN
        self.gpio_instance = lgpio.gpiochip_open(0)



    def check_play(self, status):
        if not self.use_buzzer:
            return 0
        
        if status>=7:
            self.check_play()
        else:
            self.stop()

    def check_play(self,frequency:int=2500):
        lgpio.tx_pwm(self.gpio_instance, self.PIN, frequency , 50)  # 50% duty cycle

    def stop(self):
        lgpio.tx_pwm(self.gpio_instance, self.PIN, 0, 0)
        lgpio.gpio_write(self.gpio_instance, self.PIN, 0)