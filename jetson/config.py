# ── CLOUD SERVER ──────────────────────────────────────────────────────────────
CLOUD_IP            = "x.x.x.x"   # replace with your cloud server IP
INFERENCE_PORT      = 5555

# ── CAMERA ────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 2
FRAME_WIDTH         = 1920
FRAME_HEIGHT        = 1080
CAMERA_FPS          = 30

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
MODEL_INPUT_SIZE    = 640          # YOLO input resolution (square)
JPEG_QUALITY        = 80           # compression quality sent over network

# ── MOTION GATE ───────────────────────────────────────────────────────────────
MOTION_THRESHOLD    = 5_000        # min changed pixels to consider "motion"
MOTION_BLUR_KSIZE   = 5            # gaussian blur kernel before diff
MOTION_DILATE_ITER  = 2            # dilation passes to fill gaps

# ── NETWORK ───────────────────────────────────────────────────────────────────
SEND_TIMEOUT_MS     = 2_000        # ZMQ send timeout in milliseconds
RECV_TIMEOUT_MS     = 5_000        # ZMQ recv timeout in milliseconds