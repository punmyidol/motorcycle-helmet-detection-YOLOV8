import logging
from config import (
    MODEL_PATH,
    INFERENCE_PORT,
    ALERT_PORT,
    ALERT_DEVICE_IP,
    DB_PATH,
    POLYGON,
    POLYGON_MOTORCYCLE,
)
from model import HelmetDetector
from server import InferenceServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    print("=" * 50)
    print("  HELMET DETECTION CLOUD SERVER")
    print("=" * 50)

    # ── LOAD MODEL ────────────────────────────────────────────────────────────
    print("[Main] Loading models...")
    model = HelmetDetector(
        vehicle_model_path = MODEL_PATH["vehicle"],
        helmet_model_path  = MODEL_PATH["helmet"],
        polygon            = POLYGON,
        polygon_motorcycle = POLYGON_MOTORCYCLE,
    )

    # ── START SERVER ──────────────────────────────────────────────────────────
    print("[Main] Starting inference server...")
    server = InferenceServer(
        model           = model,
        inference_port  = INFERENCE_PORT,
        alert_port      = ALERT_PORT,
        alert_device_ip = ALERT_DEVICE_IP,
        db_path         = DB_PATH,
    )

    # ── RUN ───────────────────────────────────────────────────────────────────
    try:
        server.run()
    finally:
        server.close()
        print("[Main] Server closed cleanly.")


if __name__ == "__main__":
    main()