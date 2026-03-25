import logging
import time

import cv2

from config import (
    CAMERA_INDEX,
    CAMERA_FPS,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MODEL_INPUT_SIZE,
    JPEG_QUALITY,
    MOTION_THRESHOLD,
    MOTION_BLUR_KSIZE,
    MOTION_DILATE_ITER,
    CLOUD_IP,
    INFERENCE_PORT,
    SEND_TIMEOUT_MS,
    RECV_TIMEOUT_MS,
)
from motion_gate import MotionGate
from preprocessor import Preprocessor
from sender import FrameSender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}")
    logger.info("Camera opened: %dx%d @ %d fps", width, height, fps)
    return cap


def main() -> None:
    print("=" * 50)
    print("  HELMET DETECTION — JETSON NANO")
    print("=" * 50)

    # ── INIT ──────────────────────────────────────────────────────────────────
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CAMERA_FPS)

    gate = MotionGate(
        threshold=MOTION_THRESHOLD,
        blur_ksize=MOTION_BLUR_KSIZE,
        dilate_iterations=MOTION_DILATE_ITER,
    )
    preprocessor = Preprocessor(
        model_input_size=MODEL_INPUT_SIZE,
        jpeg_quality=JPEG_QUALITY,
    )
    sender = FrameSender(
        cloud_ip=CLOUD_IP,
        inference_port=INFERENCE_PORT,
        send_timeout_ms=SEND_TIMEOUT_MS,
        recv_timeout_ms=RECV_TIMEOUT_MS,
    )

    # ── STATS ─────────────────────────────────────────────────────────────────
    frames_captured = 0
    frames_sent     = 0
    frames_skipped  = 0

    logger.info("Starting capture loop — press Ctrl-C to stop")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to read frame — retrying in 1 s")
                time.sleep(1.0)
                gate.reset()
                continue

            frames_captured += 1

            # ── MOTION GATE ───────────────────────────────────────────────────
            if not gate.has_motion(frame):
                frames_skipped += 1
                if frames_skipped % 300 == 0:
                    logger.info(
                        "Stats → captured: %d | sent: %d | skipped: %d",
                        frames_captured, frames_sent, frames_skipped,
                    )
                continue

            # ── PREPROCESS ────────────────────────────────────────────────────
            try:
                jpeg_bytes = preprocessor.process(frame)
            except ValueError as exc:
                logger.error("Preprocessing failed: %s", exc)
                continue

            # ── SEND ──────────────────────────────────────────────────────────
            if sender.send(jpeg_bytes):
                frames_sent += 1
                logger.debug("Frame sent (%d bytes)", len(jpeg_bytes))
            else:
                logger.warning("Frame dropped (send failed)")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")

    finally:
        cap.release()
        sender.close()
        logger.info(
            "Shutdown complete. Total → captured: %d | sent: %d | skipped: %d",
            frames_captured, frames_sent, frames_skipped,
        )


if __name__ == "__main__":
    main()