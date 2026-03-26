from pathlib import Path
import numpy as np

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_PATH = {
    "vehicle": Path(__file__).parent / "weights" / "yolov8m.pt",
    "helmet":  Path(__file__).parent / "weights" / "helmet-detection-v11m.pt",
}

# ── NETWORK ───────────────────────────────────────────────────────────────────
INFERENCE_PORT  = 5555
ALERT_PORT      = 5556
ALERT_DEVICE_IP = "x.x.x.x"   # replace with your alert device IP

# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "detections.db"

# ── FRAME RESOLUTION ──────────────────────────────────────────────────────────
# Must match what the Jetson sends after cropping and resizing.
# Jetson crops to square then resizes to MODEL_INPUT_SIZE × MODEL_INPUT_SIZE.
FRAME_WIDTH  = 640
FRAME_HEIGHT = 640

# ── DETECTION ZONE ────────────────────────────────────────────────────────────
# Normalized coordinates (0.0 - 1.0) of the outer detection polygon.
# polygon_motorcycle removed — the Jetson pre-crops the frame to this zone,
# so a single polygon check on the 640×640 image is sufficient.

_POLYGON_NORMALIZED = np.array([
    [1,    1   ],  # bottom right
    [0.55, 0.6 ],  # upper middle
    [0.55, 0   ],  # top middle
    [0,    0   ],  # top left
    [0,    1   ],  # bottom left
    [0.94, 1   ],  # bottom right
], dtype=np.float32)


def _to_pixel(normalized, width, height):
    """Convert normalized (0-1) polygon coords to pixel coords for OpenCV."""
    pixel = normalized.copy()
    pixel[:, 0] = (normalized[:, 0] * width).astype(np.int32)
    pixel[:, 1] = (normalized[:, 1] * height).astype(np.int32)
    return np.ascontiguousarray(pixel.reshape((-1, 1, 2)).astype(np.int32))


POLYGON = _to_pixel(_POLYGON_NORMALIZED, FRAME_WIDTH, FRAME_HEIGHT)