import lgpio
import time

BUZZER_PIN = 4
FREQ = 2500  # Frekvens i Hz

# Öppna GPIO-chip
h = lgpio.gpiochip_open(0)

# Sätt pin som output
lgpio.gpio_claim_output(h, BUZZER_PIN)

# Starta PWM
lgpio.tx_pwm(h, BUZZER_PIN, FREQ, 50)  # 50% duty cycle

time.sleep(2)  # Spela ljud i 2 sekunder

# Stoppa buzzer
lgpio.tx_pwm(h, BUZZER_PIN, 0, 0)
lgpio.gpio_write(h, BUZZER_PIN, 0)


# Stäng GPIO-chip
lgpio.gpiochip_close(h)

