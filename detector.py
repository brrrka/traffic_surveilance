"""
Vehicle detector for ANCHOR VI.
Scope: car-only detection using a COCO-pretrained YOLO11 model + ByteTrack.

Compatibility:
- Keeps VehicleDetector(model_path)
- Keeps detect_and_track(frame)
- Preserves output keys: id, bbox, class, confidence, center
- Adds class_id and footpoint for downstream analysis

ROI policy:
- ROI is OPTIONAL and disabled by default.
- Detection/tracking runs on the full frame by default.
- If ROI is enabled, filtering is applied AFTER tracking using the
  bottom-centre (footpoint) of the bounding box. This avoids cropping
  the image before tracking and keeps tracker observations available.
"""

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    # COCO class ID for "car" in standard pretrained YOLO models.
    CAR_CLASS_ID = 2

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        iou: float = 0.50,
        tracker: str = "bytetrack.yaml",
        image_size: Optional[int] = None,
        roi_points: Optional[Sequence[Tuple[int, int]]] = None,
        roi_enabled: bool = False,
    ):
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence harus berada pada rentang (0, 1].")
        if not 0.0 < iou <= 1.0:
            raise ValueError("iou harus berada pada rentang (0, 1].")

        self.model = YOLO(model_path)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.tracker = tracker
        self.image_size = image_size

        self.vehicle_classes = {self.CAR_CLASS_ID: "car"}

        self.roi_enabled = bool(roi_enabled)
        self.roi_points = self._validate_roi(roi_points)

        if self.roi_enabled and self.roi_points is None:
            raise ValueError(
                "ROI diaktifkan tetapi roi_points belum diberikan. "
                "Berikan poligon ROI atau set roi_enabled=False."
            )

    @staticmethod
    def _validate_roi(
        points: Optional[Sequence[Tuple[int, int]]]
    ) -> Optional[np.ndarray]:
        if points is None:
            return None

        polygon = np.asarray(points, dtype=np.int32)

        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError(
                "roi_points harus berupa minimal 3 titik (x, y), "
                "contoh [(x1,y1), (x2,y2), (x3,y3)]."
            )

        return polygon.reshape((-1, 1, 2))

    def _inside_roi(self, point: Tuple[int, int]) -> bool:
        """Check whether a point lies inside/on the configured ROI polygon."""
        if not self.roi_enabled or self.roi_points is None:
            return True

        return cv2.pointPolygonTest(
            self.roi_points,
            (float(point[0]), float(point[1])),
            False
        ) >= 0

    def detect_and_track(self, frame: np.ndarray):
        """
        Detect and track cars in one frame.

        Returns a list of dictionaries compatible with the existing
        TrajectoryManager and Visualizer.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Frame kosong atau bukan numpy array yang valid.")

        track_kwargs = {
            "source": frame,
            "persist": True,
            "tracker": self.tracker,
            "classes": [self.CAR_CLASS_ID],
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }

        if self.image_size is not None:
            track_kwargs["imgsz"] = int(self.image_size)

        results = self.model.track(**track_kwargs)

        if not results:
            return []

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        tracked_objects = []
        frame_height, frame_width = frame.shape[:2]

        # Move tensor data to CPU once before iterating.
        xyxy_values = boxes.xyxy.cpu().numpy()
        class_values = boxes.cls.cpu().numpy().astype(int)
        confidence_values = boxes.conf.cpu().numpy()
        track_ids = boxes.id.int().cpu().tolist()

        for bbox, cls_id, confidence, track_id in zip(
            xyxy_values, class_values, confidence_values, track_ids
        ):
            if cls_id != self.CAR_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            # Clamp coordinates to the image boundaries.
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Bounding-box centre is retained for compatibility.
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Bottom-centre is a more useful ground-contact approximation
            # for road-plane trajectory and optional ROI membership.
            foot_x = cx
            foot_y = y2

            if not self._inside_roi((foot_x, foot_y)):
                continue

            tracked_objects.append({
                "id": int(track_id),
                "bbox": [x1, y1, x2, y2],
                "class": "car",
                "class_id": int(cls_id),
                "confidence": float(confidence),
                "center": (cx, cy),
                "footpoint": (foot_x, foot_y),
            })

        return tracked_objects

    def reset_tracker(self):
        """
        Reset tracker state when starting a completely new video with the
        same VehicleDetector instance, if supported by the installed
        Ultralytics version.
        """
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)

        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()
