import cv2
import numpy as np

from config import MOTION_BLUR_KERNEL, MOTION_DIFF_THRESH, MOTION_THRESHOLD


class MotionGate(object):
    """
    Frame-differencing motion detector.

    Compares each incoming frame against the previous one.
    Returns True (motion detected) or False (scene is static, skip frame).

    Conservative tuning: only skips frames when the scene is genuinely still.
    Raise MOTION_THRESHOLD in config.py to make gating more aggressive.
    """

    def __init__(self):
        self._prev_gray = None  # grayscale of the last seen frame

    def has_motion(self, frame):
        """
        Parameters
        ----------
        frame : numpy.ndarray
            Full BGR frame from cv2.VideoCapture.

        Returns
        -------
        bool
            True  -- motion detected, caller should process/send this frame.
            False -- scene is static, caller should skip this frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (MOTION_BLUR_KERNEL, MOTION_BLUR_KERNEL), 0)

        # First frame -- no previous reference yet, always pass through
        if self._prev_gray is None:
            self._prev_gray = gray
            return True

        diff        = cv2.absdiff(self._prev_gray, gray)
        _, thresh   = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        changed_px  = cv2.countNonZero(thresh)

        self._prev_gray = gray  # update reference every frame regardless

        return changed_px >= MOTION_THRESHOLD

    def reset(self):
        """Clear stored reference frame (call after a long pause/reconnect)."""
        self._prev_gray = None