import zmq
import logging

from config import ALERT_PORT, LOG_PATH
from cooldown import CooldownTimer
from alert import trigger_alert
from message_schema import is_valid_label, should_alert

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
# Writes to both the console and a local log file simultaneously.
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_file_handler = logging.FileHandler(LOG_PATH)
_file_handler.setFormatter(_formatter)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)

logger = logging.getLogger("alert-device")
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)


class AlertListener(object):
    """
    Listens on a ZeroMQ PULL socket for detection labels from the cloud.

    On every received label:
        - Validates it against the message schema
        - Logs it regardless of what it is
        - If label is "no-helmet" AND cooldown has expired -> triggers alert
    """

    def __init__(self):
        self._cooldown = CooldownTimer()

        self._context = zmq.Context()
        self._socket  = self._context.socket(zmq.PULL)
        self._socket.bind("tcp://*:%d" % ALERT_PORT)
        logger.info("PULL socket bound on port %d", ALERT_PORT)

    def run(self):
        """Block forever, processing one label per loop iteration."""
        logger.info("Alert listener ready, waiting for detections...")

        while True:
            try:
                self._handle_message()
            except KeyboardInterrupt:
                print("\n[Listener] Shutting down...")
                break
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))

    def _handle_message(self):
        label = self._socket.recv_string()

        # ── VALIDATE ──────────────────────────────────────────────────────────
        if not is_valid_label(label):
            logger.warning("Received unknown label: '%s', ignoring", label)
            return

        # ── LOG EVERY DETECTION ───────────────────────────────────────────────
        logger.info("Detection received: %s", label)

        # ── ALERT LOGIC ───────────────────────────────────────────────────────
        if should_alert(label):
            if self._cooldown.is_ready():
                logger.warning("NO-HELMET detected -- triggering alert")
                trigger_alert()
                self._cooldown.trip()
            else:
                logger.info(
                    "NO-HELMET detected -- suppressed (cooldown %.1fs remaining)",
                    self._cooldown.remaining()
                )

    def close(self):
        """Release ZeroMQ resources."""
        self._socket.close()
        self._context.term()
        logger.info("Listener closed.")