import sys
import time
import cv2

from config import (
    CAMERA_SOURCE,
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    CAPTURE_FPS,
)
from motion_gate import MotionGate
from preprocessor import Preprocessor
from sender import FrameSender
from capture_props import get_capture_props


def _set_prop(cap, cv3_const, cv2_const, value):
    """Set a VideoCapture property, falling back to the OpenCV 2.x constant."""
    try:
        cap.set(cv3_const, value)
    except AttributeError:
        cap.set(cv2_const, value)


def open_camera(source):
    """Open camera, apply resolution/FPS hints, and log actual properties."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[Main] ERROR: Cannot open camera source: %s" % str(source))
        sys.exit(1)

    _set_prop(cap, cv2.CAP_PROP_FRAME_WIDTH,  cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    _set_prop(cap, cv2.CAP_PROP_FRAME_HEIGHT, cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    _set_prop(cap, cv2.CAP_PROP_FPS,          cv2.cv.CV_CAP_PROP_FPS,          CAPTURE_FPS)

    actual_w, actual_h, actual_fps = get_capture_props(cap)
    print("[Main] Camera opened: %dx%d @ %d fps" % (actual_w, actual_h, actual_fps))
    return cap


def main():
    print("=" * 50)
    print("  HELMET DETECTION -- JETSON CAPTURE NODE")
    print("=" * 50)

    cap          = open_camera(CAMERA_SOURCE)
    gate         = MotionGate()
    preprocessor = Preprocessor()
    sender       = FrameSender()

    frames_read    = 0
    frames_sent    = 0
    frames_skipped = 0
    t_start        = time.time()

    print("[Main] Capture loop started. Press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Main] WARNING: Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            frames_read += 1

            # -- MOTION GATE --------------------------------------------------
            if not gate.has_motion(frame):
                frames_skipped += 1
                continue

            # -- PREPROCESS ---------------------------------------------------
            jpeg_bytes = preprocessor.process(frame)
            if jpeg_bytes is None:
                print("[Main] WARNING: JPEG encode failed, skipping frame")
                continue

            # -- SEND ---------------------------------------------------------
            success = sender.send(jpeg_bytes)
            if success:
                frames_sent += 1
            else:
                print("[Main] WARNING: Frame not acknowledged by cloud")

            # -- STATS (every 100 frames read) --------------------------------
            if frames_read % 100 == 0:
                elapsed  = time.time() - t_start
                fps_read = frames_read  / elapsed
                fps_sent = frames_sent  / elapsed
                skip_pct = 100.0 * frames_skipped / frames_read
                print("[Main] read=%.1f fps  sent=%.1f fps  skipped=%.1f%%" % (
                    fps_read, fps_sent, skip_pct))

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user, shutting down...")

    finally:
        cap.release()
        sender.close()

        elapsed = time.time() - t_start
        print("[Main] Session summary:")
        print("  Total frames read : %d" % frames_read)
        print("  Frames sent       : %d" % frames_sent)
        print("  Frames skipped    : %d" % frames_skipped)
        print("  Runtime           : %.1f s" % elapsed)
        print("[Main] Clean shutdown complete.")


if __name__ == "__main__":
    main()