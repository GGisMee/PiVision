import lgpio
import time
class BuzzerManager:
    def __init__(self, PIN:int = 4,use_buzzer:bool=False):
        self.use_buzzer = use_buzzer
        self.PIN = PIN
        self.gpio_instance = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.gpio_instance, self.PIN)




    def check_play(self, status):
        if not self.use_buzzer:
            return 0
        
        if status>=7:
            self.play()
        else:
            self.stop()

    def play(self,frequency:int=2500):
        lgpio.tx_pwm(self.gpio_instance, self.PIN, frequency , 90)  # 50% duty cycle

    def stop(self):
        lgpio.tx_pwm(self.gpio_instance, self.PIN, 0, 0)
        lgpio.gpio_write(self.gpio_instance, self.PIN, 0)


if __name__ == '__main__':
    buzzer_manager = BuzzerManager(use_buzzer=True)
    buzzer_manager.play()
    time.sleep(2)
    buzzer_manager.stop()