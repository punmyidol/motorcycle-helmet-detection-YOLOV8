import sys

from config import USE_RPI_GPIO, GPIO_LED_PIN, GPIO_BUZZER_PIN

# ── GPIO SETUP ────────────────────────────────────────────────────────────────
# Conditionally import the right GPIO library depending on the platform.
# Both RPi.GPIO and Jetson.GPIO share the same API, so the rest of the
# code below works unchanged on either board.

_gpio = None  # will be set to the GPIO module if available

try:
    if USE_RPI_GPIO:
        import RPi.GPIO as GPIO
    else:
        import Jetson.GPIO as GPIO

    _gpio = GPIO
    _gpio.setmode(_gpio.BCM)
    _gpio.setup(GPIO_LED_PIN,    _gpio.OUT, initial=_gpio.LOW)
    _gpio.setup(GPIO_BUZZER_PIN, _gpio.OUT, initial=_gpio.LOW)
    print("[Alert] GPIO initialised (BCM mode). LED pin=%d, Buzzer pin=%d" % (
        GPIO_LED_PIN, GPIO_BUZZER_PIN))

except ImportError:
    print("[Alert] WARNING: GPIO library not found. Hardware outputs disabled.")
except Exception as e:
    print("[Alert] WARNING: GPIO setup failed: %s" % str(e))


# =============================================================================
# PUBLIC INTERFACE
# Fill in the body of each function when you're ready to wire the hardware.
# The listener calls only trigger_alert() and cleanup().
# =============================================================================

def trigger_alert():
    """
    Called once per detection event (after cooldown check).
    Add buzzer, LED, relay, audio, or any other output here.
    """
    _trigger_led()
    _trigger_buzzer()
    _trigger_audio()


def cleanup():
    """Release GPIO resources on shutdown."""
    if _gpio is not None:
        try:
            _gpio.cleanup()
            print("[Alert] GPIO cleaned up.")
        except Exception as e:
            print("[Alert] WARNING: GPIO cleanup error: %s" % str(e))


# =============================================================================
# HARDWARE STUBS
# Each function is intentionally empty -- implement when hardware is wired.
# =============================================================================

def _trigger_led():
    """Flash the LED on GPIO_LED_PIN."""
    # TODO: implement LED flash, e.g.:
    # if _gpio:
    #     _gpio.output(GPIO_LED_PIN, _gpio.HIGH)
    #     time.sleep(0.5)
    #     _gpio.output(GPIO_LED_PIN, _gpio.LOW)
    pass


def _trigger_buzzer():
    """Sound the buzzer / activate relay on GPIO_BUZZER_PIN."""
    # TODO: implement buzzer pulse, e.g.:
    # if _gpio:
    #     _gpio.output(GPIO_BUZZER_PIN, _gpio.HIGH)
    #     time.sleep(0.3)
    #     _gpio.output(GPIO_BUZZER_PIN, _gpio.LOW)
    pass


def _trigger_audio():
    """Play an audio alert via pygame or aplay."""
    # TODO: implement audio, e.g.:
    # import pygame
    # pygame.mixer.Sound("alert.wav").play()
    # or:
    # import os
    # os.system("aplay alert.wav")
    pass