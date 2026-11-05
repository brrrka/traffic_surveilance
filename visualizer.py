"""ANCHOR VI overlay; displays km/h only when a metric/labelled prior exists."""
import cv2
import numpy as np

try:
    from config import ROI_POINTS
except Exception:
    ROI_POINTS = None


class Visualizer:
    def draw_all(self, frame, tracked_objects, histories, accidents=None, frame_number=None, fps=None):
        if ROI_POINTS:
            try:
                cv2.polylines(frame, [np.asarray(ROI_POINTS, dtype=np.int32)], True, (255, 0, 0), 2)
            except Exception:
                pass

        for obj in tracked_objects or []:
            try:
                track_id = int(obj["id"])
                x1, y1, x2, y2 = map(int, obj["bbox"])
            except Exception:
                continue
            history = histories.get(track_id, {})
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            confidence = obj.get("confidence")
            base = f"Car ID {track_id}"
            if confidence is not None:
                base += f" | YOLO {float(confidence):.2f}"
            cv2.putText(frame, base, (x1, max(18, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 2)

            speed_kmh = history.get("speed_kmh")
            speed_px = history.get("speed")
            status = history.get("speed_status", "uncalibrated")
            if speed_kmh is not None:
                marker = "~" if status == "estimated_prior" else ""
                speed_text = f"{marker}{float(speed_kmh):.1f} km/h"
                color = (0, 190, 255) if status == "estimated_prior" else (255, 220, 0)
            elif speed_px is not None:
                speed_text = f"{float(speed_px):.1f} px/s"
                color = (180, 180, 180)
            else:
                speed_text = "speed pending"
                color = (180, 180, 180)
            cv2.putText(frame, speed_text, (x1, min(frame.shape[0] - 8, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            points = history.get("trajectory") or []
            for i in range(1, len(points)):
                try:
                    p1 = tuple(map(int, points[i - 1]))
                    p2 = tuple(map(int, points[i]))
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)
                except Exception:
                    continue

        for event in accidents or []:
            loc = event.get("location")
            if loc:
                loc = tuple(map(int, loc))
                cv2.circle(frame, loc, 28, (0, 0, 255), 2)
                cv2.circle(frame, loc, 6, (0, 0, 255), -1)
            label = event.get("crash_type") or "Kandidat kontak"
            angle = event.get("impact_angle_deg")
            if angle is not None:
                label += f" | approach angle {float(angle):.1f} deg"
            cv2.putText(frame, label[:100], (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        if frame_number is not None:
            cv2.putText(frame, f"ANCHOR VI | Frame {int(frame_number)}", (12, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1)
        return frame
