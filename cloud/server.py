import zmq
import cv2
import numpy as np
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class InferenceServer:
    """
    ZeroMQ networking layer for the helmet detection cloud server.

    Sockets
    -------
        REP on INFERENCE_PORT  — receives JPEG frames from Jetson,
                                  sends detection label back as ACK
        PUSH on ALERT_PORT     — sends alert label to alert device

    Database
    --------
        SQLite — saves every processed annotated frame as BLOB
    """

    def __init__(self, model, inference_port, alert_port, alert_device_ip, db_path):
        self.model = model

        # ── ZEROMQ ────────────────────────────────────────────────────────────
        self.context = zmq.Context()

        # REP socket — receive frame from Jetson, send label back as ACK
        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://*:{inference_port}")
        logger.info(f"REP socket bound on port {inference_port}")

        # PUSH socket — send alert label to alert device
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://{alert_device_ip}:{alert_port}")
        logger.info(f"PUSH socket connected to {alert_device_ip}:{alert_port}")

        # ── DATABASE ──────────────────────────────────────────────────────────
        self.db_path = db_path
        self._init_db()

    def run(self):
        """Block forever, processing one frame per loop iteration."""
        logger.info("Server ready, waiting for frames...")
        print("[Server] Listening for incoming frames...")

        while True:
            try:
                self._handle_frame()
            except KeyboardInterrupt:
                print("[Server] Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error handling frame: {e}")
                # Send unknown so Jetson REQ socket doesn't hang
                self.rep_socket.send_string("unknown")

    def _handle_frame(self):
        # ── RECEIVE ───────────────────────────────────────────────────────────
        jpeg_bytes = self.rep_socket.recv()
        frame = self._decode(jpeg_bytes)

        if frame is None:
            logger.warning("Failed to decode incoming frame, skipping")
            self.rep_socket.send_string("unknown")
            return

        # ── INFERENCE ─────────────────────────────────────────────────────────
        result       = self.model.run(frame)
        alert_label  = result["alert_label"]        # "helmet" | "no-helmet" | "unknown"
        annotated    = result["annotated_frame"]

        logger.info(f"Detection result: {alert_label}")

        # ── SAVE TO DATABASE ──────────────────────────────────────────────────
        self._save_frame(annotated, alert_label)

        # ── PUSH LABEL TO ALERT DEVICE ────────────────────────────────────────
        self.push_socket.send_string(alert_label)

        # ── ACK BACK TO JETSON (label as response) ────────────────────────────
        self.rep_socket.send_string(alert_label)

    # ── DATABASE ──────────────────────────────────────────────────────────────

    def _init_db(self):
        """Create the detections table if it doesn't exist yet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT    NOT NULL,
                    label     TEXT    NOT NULL,
                    frame     BLOB    NOT NULL
                )
            """)
            conn.commit()
        logger.info(f"Database ready at {self.db_path}")

    def _save_frame(self, frame, label):
        """Encode annotated frame as JPEG and save to database."""
        jpeg_bytes = self._encode(frame)
        timestamp  = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO detections (timestamp, label, frame) VALUES (?, ?, ?)",
                (timestamp, label, sqlite3.Binary(jpeg_bytes))
            )
            conn.commit()

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _decode(self, jpeg_bytes):
        """JPEG bytes → numpy BGR frame."""
        buf   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return frame  # None if decode failed

    def _encode(self, frame):
        """numpy BGR frame → JPEG bytes."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    def close(self):
        """Clean up ZeroMQ and database resources."""
        self.rep_socket.close()
        self.push_socket.close()
        self.context.term()