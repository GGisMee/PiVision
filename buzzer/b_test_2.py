import lgpio
import time

BUZZER_PIN = 3
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, BUZZER_PIN)

try:
    print("Playing horn sound...")
    for freq in range(300, 800, 10):  # Gradually increase pitch
        lgpio.tx_pwm(h, BUZZER_PIN, freq, 80)  # 80% duty cycle
        time.sleep(0.1)

    for freq in range(800, 300, -10):  # Gradually decrease pitch
        lgpio.tx_pwm(h, BUZZER_PIN, freq, 80)
        time.sleep(0.1)

finally:
    lgpio.gpiochip_close(h)


