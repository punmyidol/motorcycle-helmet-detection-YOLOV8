import time

from config import COOLDOWN_SECONDS


class CooldownTimer(object):
    """
    Prevents the alert from firing repeatedly for the same incident.

    Once tripped, is_ready() returns False until COOLDOWN_SECONDS have elapsed.
    """

    def __init__(self):
        self._last_alert_time = 0  # epoch seconds; 0 means never fired

    def is_ready(self):
        """Return True if enough time has passed since the last alert."""
        return (time.time() - self._last_alert_time) >= COOLDOWN_SECONDS

    def trip(self):
        """Record that an alert just fired, starting the cooldown window."""
        self._last_alert_time = time.time()

    def reset(self):
        """Manually clear the cooldown (useful for testing)."""
        self._last_alert_time = 0

    def remaining(self):
        """Return seconds left in the current cooldown window (0 if ready)."""
        elapsed = time.time() - self._last_alert_time
        remaining = COOLDOWN_SECONDS - elapsed
        return max(0.0, remaining)