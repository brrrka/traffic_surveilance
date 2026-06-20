from ultralytics.trackers.byte_tracker import BYTETracker
from types import SimpleNamespace

class VehicleTracker:
    def __init__(self):
        config = SimpleNamespace(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=30
        )
        self.tracker = BYTETracker(args=config)

    def update(self, detections):
        results = []
        tracker_input = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            tracker_input.append([x1, y1, x2, y2, conf])

        if len(tracker_input) == 0:
            return []

        tracks = self.tracker.update(tracker_input)

        for track in tracks:
            track_id = int(track.track_id)
            x1, y1, x2, y2 = map(int, track.tlbr)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            results.append({
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy)
            })

        return results