import cv2
import numpy as np


POLYGON_NORMALIZED = np.array([
    [1,    1   ],  # bottom right
    [0.55, 0.6 ],  # upper middle
    [0.55, 0   ],  # top middle
    [0,    0   ],  # top left
    [0,    1   ],  # bottom left
    [0.94, 1   ],  # bottom right
], dtype=np.float32)


class Preprocessor:
    """
    Crops the frame to the tightest square that fits inside the detection
    polygon, then resizes to model_input_size × model_input_size and
    JPEG-compresses for network transmission.

    Cropping on the Jetson means the cloud receives a properly framed
    image where polygon coords are always valid — no horizontal compression.
    """

    def __init__(self, model_input_size: int = 640, jpeg_quality: int = 80) -> None:
        self._size          = model_input_size
        self._quality       = jpeg_quality
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]

        # Pixel coords are computed lazily on the first frame so we don't
        # need to know the resolution at construction time.
        self._last_shape = None
        self._crop_box   = None  # (x1, y1, x2, y2) in pixel coords

    def process(self, frame: np.ndarray) -> bytes:
        """
        Crop to detection zone → resize to square → JPEG encode.

        Args:
            frame: BGR numpy array from cv2.VideoCapture.

        Returns:
            JPEG-encoded bytes ready to send over the network.

        Raises:
            ValueError: if the crop region is empty or JPEG encoding fails.
        """
        h, w = frame.shape[:2]

        # Recompute crop box only when resolution changes (e.g. first frame)
        if (w, h) != self._last_shape:
            self._crop_box   = self._compute_crop_box(w, h)
            self._last_shape = (w, h)

        x1, y1, x2, y2 = self._crop_box
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            raise ValueError(
                f"Crop region is empty — box ({x1},{y1})-({x2},{y2}) "
                f"is outside frame {w}×{h}"
            )

        resized = cv2.resize(
            cropped,
            (self._size, self._size),
            interpolation=cv2.INTER_LINEAR,
        )

        ok, buf = cv2.imencode(".jpg", resized, self._encode_params)
        if not ok:
            raise ValueError("JPEG encoding failed — frame may be corrupted")

        return buf.tobytes()

    def get_crop_box(self, frame_width: int, frame_height: int):
        """
        Return the (x1, y1, x2, y2) pixel crop box for a given resolution.
        Useful for debugging or drawing the crop region on a display frame.
        """
        return self._compute_crop_box(frame_width, frame_height)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _compute_crop_box(self, w: int, h: int):
        """
        Convert the normalized polygon to pixel coords, find its axis-aligned
        bounding box, then shrink to the largest square that fits inside it
        without going outside the frame.

        Strategy
        --------
        The polygon bbox gives us (x_min, y_min, x_max, y_max).
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        We want a square of side = min(bbox_w, bbox_h) so we never reach
        outside the polygon bbox. The square is anchored at (x_min, y_min).
        """
        # Scale normalized → pixel
        pts       = POLYGON_NORMALIZED.copy()
        pts[:, 0] = pts[:, 0] * w
        pts[:, 1] = pts[:, 1] * h
        pts       = pts.astype(np.int32)

        x_min = int(pts[:, 0].min())
        y_min = int(pts[:, 1].min())
        x_max = int(pts[:, 0].max())
        y_max = int(pts[:, 1].max())

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Largest square that fits inside the bounding box
        side = min(bbox_w, bbox_h)

        x1 = x_min
        y1 = y_min
        x2 = x1 + side
        y2 = y1 + side

        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return x1, y1, x2, y2