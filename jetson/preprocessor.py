import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Resizes a raw camera frame to the model's expected input size and
    JPEG-compresses it for efficient network transmission.

    Doing both steps on the Jetson — before sending — keeps the cloud
    server's decode path simple and minimises bandwidth usage.
    """

    def __init__(self, model_input_size: int = 640, jpeg_quality: int = 80) -> None:
        self._size    = model_input_size
        self._quality = jpeg_quality
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]

    def process(self, frame: np.ndarray) -> bytes:
        """
        Resize *frame* to a square (model_input_size × model_input_size)
        and return the result as compressed JPEG bytes.

        Args:
            frame: BGR numpy array from cv2.VideoCapture.

        Returns:
            JPEG-encoded bytes ready to send over the network.

        Raises:
            ValueError: if JPEG encoding fails (e.g. corrupted frame).
        """
        resized = cv2.resize(
            frame,
            (self._size, self._size),
            interpolation=cv2.INTER_LINEAR,
        )

        ok, buf = cv2.imencode(".jpg", resized, self._encode_params)
        if not ok:
            raise ValueError("JPEG encoding failed — frame may be corrupted")

        return buf.tobytes()