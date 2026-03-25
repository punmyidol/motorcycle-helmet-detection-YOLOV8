import cv2


def _prop(cap, cv3_const, cv2_const):
    """
    Get a VideoCapture property, falling back to the OpenCV 2.x constant
    name if the OpenCV 3.x name is not available.
    """
    try:
        val = cap.get(cv3_const)
    except AttributeError:
        val = cap.get(cv2_const)
    return val


def get_capture_props(cap):
    """
    Read width, height, and FPS from an open cv2.VideoCapture.

    Works on Python 2.7 with OpenCV 2.x and 3.x.

    Returns
    -------
    (width, height, fps) -- all ints, with safe fallbacks if the driver
    does not report a value (common with USB webcams on Linux).
    """
    cap.set(3, 1920)  # width
    cap.set(4, 1080)  # height
    cap.set(5, 30)    # fps

    # USB webcam drivers on Linux often report 0 -- fall back to safe defaults
    if width  == 0: width  = 1920
    if height == 0: height = 1080
    if fps    == 0: fps    = 30

    return width, height, fps


# ── USAGE ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open camera")
    else:
        width, height, fps = get_capture_props(cap)
        print("Width  : %d" % width)
        print("Height : %d" % height)
        print("FPS    : %d" % fps)
        cap.release()