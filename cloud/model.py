import cv2
import math
import numpy as np
from ultralytics import YOLO

MOTORCYCLE_CLASS = 3
BICYCLE_CLASS    = 1
HELMET_WITH      = 0
HELMET_WITHOUT   = 1

VEHICLE_CONFIDENCE = 0.3
HELMET_CONFIDENCE  = 0.35   # matches helmet_model_cls.py
EXPANSION_FACTOR   = 0.6    # matches helmet_model_cls.py
OVERLAP_THRESHOLD  = 7


class HelmetDetector:

    def __init__(self, vehicle_model_path, helmet_model_path, polygon):
        self.vehicle_model = YOLO(vehicle_model_path, verbose=False)
        self.helmet_model  = YOLO(helmet_model_path,  verbose=False)
        self.polygon       = polygon
        # polygon_motorcycle removed — frame is pre-cropped by Jetson

    def run(self, frame):
        """
        Returns a dict:
            should_alert    : bool
            alert_label     : "helmet" | "no-helmet" | "unknown"
            annotated_frame : frame with boxes drawn
        """
        vehicle_detections = self._detect_vehicles(frame)
        results, annotated = self._detect_helmets(frame, vehicle_detections)
        alert_label        = self._determine_alert(results)

        return {
            "should_alert":    alert_label == "no-helmet",
            "alert_label":     alert_label,
            "annotated_frame": annotated,
        }

    # ── STAGE 1: VEHICLE ─────────────────────────────────────────────────────

    def _detect_vehicles(self, frame):
        """
        Run vehicle model, keep only motorcycles and bicycles whose centre
        falls inside the detection polygon.
        Matches helmet_model_cls._detect_vehicles() + polygon filter.
        """
        raw = self.vehicle_model.predict(frame, conf=VEHICLE_CONFIDENCE, verbose=False)
        detections = []

        for r in raw:
            for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                cls  = int(cls)
                conf = float(conf)

                if cls not in (MOTORCYCLE_CLASS, BICYCLE_CLASS):
                    continue

                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2

                # Single polygon check — frame is already cropped to zone
                if cv2.pointPolygonTest(self.polygon, (float(cx), float(cy)), False) < 0:
                    continue

                detections.append((x1, y1, x2, y2, conf, cls))

        return detections

    # ── STAGE 2: HELMET ──────────────────────────────────────────────────────

    def _detect_helmets(self, frame, vehicle_detections):
        """
        For each motorcycle, run the helmet model on the top half of the
        frame (riders/heads are always in the upper portion of a cropped
        frame). Matches helmet_model_cls._process_motorcycle() logic.
        """
        h, w      = frame.shape[:2]
        annotated = frame.copy()
        results   = []

        for x1, y1, x2, y2, vconf, cls in vehicle_detections:

            if cls == BICYCLE_CLASS:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                continue  # no helmet check for bicycles

            # ── Draw motorcycle box ───────────────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ── Build ROI: expanded box clipped to top half of frame ──────────
            # Expand bounding box to capture the rider above the motorcycle.
            # Matches helmet_model_cls expansion_factor=0.6.
            bw = x2 - x1
            bh = y2 - y1

            ex1 = max(0, int(x1 - bw * EXPANSION_FACTOR * 0.5))
            ey1 = max(0, int(y1 - bh * EXPANSION_FACTOR * 1.5))
            ex2 = min(w, int(x2 + bw * EXPANSION_FACTOR * 0.5))
            ey2 = h // 2  # top half of frame only

            # Skip if the motorcycle is entirely in the bottom half
            if ey1 >= ey2:
                continue

            roi = frame[ey1:ey2, ex1:ex2]
            if roi.size == 0:
                continue

            # ── Run helmet model on ROI ───────────────────────────────────────
            # Matches helmet_model_cls._detect_helmets_in_roi()
            raw_helmets = []
            for det in self.helmet_model.predict(roi, conf=HELMET_CONFIDENCE, verbose=False):
                for b in det.boxes:
                    hx1, hy1, hx2, hy2 = map(int, b.xyxy[0].tolist())
                    hconf = float(b.conf[0])
                    hcls  = int(b.cls[0])
                    # Remap ROI coords → full frame coords
                    raw_helmets.append((
                        hx1 + ex1, hy1 + ey1,
                        hx2 + ex1, hy2 + ey1,
                        hconf, hcls,
                    ))

            helmets = self._filter_overlapping(raw_helmets, threshold=OVERLAP_THRESHOLD)

            # ── Classify and annotate ─────────────────────────────────────────
            for hx1, hy1, hx2, hy2, hconf, hcls in helmets:
                hcx = hx1 + (hx2 - hx1) // 2
                hcy = hy1 + (hy2 - hy1) // 2

                # Helmet centre must still be inside detection zone
                if cv2.pointPolygonTest(self.polygon, (float(hcx), float(hcy)), False) < 0:
                    continue

                if hcls == HELMET_WITH:
                    results.append("helmet")
                    cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                    cv2.putText(annotated,
                                "With Helmet {:.2f}".format(hconf),
                                (max(0, hx1), max(35, hy1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                elif hcls == HELMET_WITHOUT:
                    results.append("no-helmet")
                    cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                    cv2.putText(annotated,
                                "Without Helmet {:.2f}".format(hconf),
                                (max(0, hx1), max(35, hy1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return results, annotated

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _determine_alert(self, results):
        """
        no-helmet takes priority over helmet.
        Returns unknown only when no motorcycle produced any helmet detection.
        Matches helmet_model_cls implicit logic (any without-helmet → alert).
        """
        if "no-helmet" in results:
            return "no-helmet"
        if "helmet" in results:
            return "helmet"
        return "unknown"

    def _filter_overlapping(self, detections, threshold=7):
        """
        Keep highest-confidence detection when two boxes have centres within
        threshold pixels. Matches helmet_model_cls.filter_overlapping_helmets().
        """
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
        filtered, centers = [], []

        for det in sorted_dets:
            x1, y1, x2, y2, conf, cls = det
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            too_close = any(
                math.sqrt((cx - fcx) ** 2 + (cy - fcy) ** 2) <= threshold
                for fcx, fcy in centers
            )
            if not too_close:
                filtered.append(det)
                centers.append((cx, cy))

        return filtered