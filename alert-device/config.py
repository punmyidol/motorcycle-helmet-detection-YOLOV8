# ── NETWORK ───────────────────────────────────────────────────────────────────
# Must match ALERT_PORT in cloud/config.py
ALERT_PORT = 5556

# ── COOLDOWN ──────────────────────────────────────────────────────────────────
# Seconds to suppress repeat alerts after one fires.
# e.g. 5 means once an alert triggers, the next 5 seconds of no-helmet
# detections are silently ignored.
COOLDOWN_SECONDS = 5

# ── LOGGING ───────────────────────────────────────────────────────────────────
import os
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.log")

# ── HARDWARE ──────────────────────────────────────────────────────────────────
# Set to True only on a Raspberry Pi with RPi.GPIO installed.
# Set to False on a Jetson Nano (uses Jetson.GPIO instead, wired in alert.py).
USE_RPI_GPIO = True     # change to False if running on Jetson

# GPIO pin numbers (BCM numbering) -- only used when GPIO is enabled
GPIO_LED_PIN    = 17    # LED output pin
GPIO_BUZZER_PIN = 27    # buzzer / relay output pin