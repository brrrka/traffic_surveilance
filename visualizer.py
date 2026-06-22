import cv2
import numpy as np
from config import ROI_POINTS

class Visualizer:
    def draw_all(self, frame, tracked_objects, histories, accidents=None):
        cv2.polylines(frame, [np.array(ROI_POINTS)], True, (255, 0, 0), 3)

        for obj in tracked_objects:
            track_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            v_class = obj.get("class", "unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{v_class} {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if track_id in histories:
                points = histories[track_id]["trajectory"]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

        if accidents:
            for acc in accidents:
                loc = acc["location"]
                text = f"CRASH: {acc['class1']} ({acc['v1']}) vs {acc['class2']} ({acc['v2']})"
                
                cv2.circle(frame, loc, 40, (0, 0, 255), 2)
                cv2.circle(frame, loc, 10, (0, 0, 255), -1)
                cv2.putText(frame, text, (loc[0] - 150, loc[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame