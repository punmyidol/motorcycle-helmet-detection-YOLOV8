import cv2
import numpy as np

from config import INFER_WIDTH, INFER_HEIGHT, JPEG_QUALITY


class Preprocessor(object):
    """
    Resizes a frame to the model input resolution and JPEG-encodes it.

    Keeping this as a class (rather than bare functions) makes it easy
    to swap in letterbox-padding later if the model requires it.
    """

    def __init__(self):
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    def process(self, frame):
        """
        Parameters
        ----------
        frame : numpy.ndarray
            Raw BGR frame, any resolution.

        Returns
        -------
        bytes or None
            JPEG-encoded bytes ready to send over the network,
            or None if encoding failed.
        """
        resized = self._resize(frame)
        return self._encode(resized)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _resize(self, frame):
        """Resize to INFER_WIDTH x INFER_HEIGHT using fast linear interpolation."""
        return cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT),
                          interpolation=cv2.INTER_LINEAR)

    def _encode(self, frame):
        """Encode numpy BGR frame to JPEG bytes."""
        ok, buf = cv2.imencode(".jpg", frame, self._encode_params)
        if not ok:
            return None
        # imencode returns a numpy array; tobytes() works on both Py2 and Py3
        return buf.tobytes()