import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MotionGate:
    """
    Frame-differencing motion detector.

    Compares the current frame against the previous one. If the number of
    significantly changed pixels exceeds ``threshold``, the frame is
    considered "active" and should be forwarded to the cloud for inference.

    Typical savings: 80-90 % reduction in frames sent when traffic is sparse.
    """

    def __init__(
        self,
        threshold: int = 5_000,
        blur_ksize: int = 5,
        dilate_iterations: int = 2,
    ) -> None:
        self._threshold        = threshold
        self._blur_ksize       = blur_ksize
        self._dilate_iters     = dilate_iterations
        self._prev_gray: np.ndarray | None = None

    def has_motion(self, frame: np.ndarray) -> bool:
        """
        Return ``True`` if meaningful motion is detected in *frame*.

        Always returns ``True`` on the very first call (no previous frame
        to compare against) so the first frame is never silently dropped.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._blur_ksize, self._blur_ksize), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return True  # treat first frame as active

        diff   = cv2.absdiff(self._prev_gray, gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Fill small gaps so a large moving object isn't fragmented
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.dilate(mask, kernel, iterations=self._dilate_iters)

        changed_pixels = int(np.count_nonzero(mask))
        self._prev_gray = gray

        motion_detected = changed_pixels >= self._threshold
        if motion_detected:
            logger.debug("Motion detected: %d changed pixels", changed_pixels)
        return motion_detected

    def reset(self) -> None:
        """Discard the stored reference frame (e.g. after a camera reconnect)."""
        self._prev_gray = None