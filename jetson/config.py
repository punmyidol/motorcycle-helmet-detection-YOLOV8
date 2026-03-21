# cloud IP, ports, thresholds

# ── CAMERA ────────────────────────────────────────────────────────────────────
# Set to an integer (0, 1, ...) for a USB webcam, or a string for an RTSP URL.
# Examples:
#   CAMERA_SOURCE = 0
#   CAMERA_SOURCE = "rtsp://user:pass@192.168.1.10:554/stream"
CAMERA_SOURCE = 0

CAPTURE_WIDTH  = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS    = 30        # target FPS for cv2.VideoCapture

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
# Resolution sent to the cloud (must match what the YOLO model expects)
INFER_WIDTH  = 640
INFER_HEIGHT = 640

JPEG_QUALITY = 80          # 0-100; 80 keeps file small without visible loss

# ── MOTION GATE ───────────────────────────────────────────────────────────────
# Conservative settings -- only skip frames that are truly static.
# Raise MOTION_THRESHOLD to skip more frames (more aggressive gating).
MOTION_THRESHOLD   = 600000   # minimum changed-pixel count to count as motion
MOTION_BLUR_KERNEL = 5     # gaussian blur kernel size (must be odd)
MOTION_DIFF_THRESH = 25    # per-pixel abs-diff value to call a pixel "changed"

# ── NETWORK ───────────────────────────────────────────────────────────────────
CLOUD_IP        = "x.x.x.x"   # replace with your cloud server IP
INFERENCE_PORT  = 5555
SEND_TIMEOUT_MS = 5000         # ZMQ send/recv timeout in milliseconds