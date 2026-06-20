from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.vehicle_classes = {
            2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
        }

    def detect_and_track(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        tracked_objects = []
        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id in zip(result.boxes, result.boxes.id):
                cls_id = int(box.cls[0])
                
                if cls_id not in self.vehicle_classes:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                tracked_objects.append({
                    "id": int(track_id),
                    "bbox": [x1, y1, x2, y2],
                    "class": self.vehicle_classes[cls_id],
                    "confidence": float(box.conf[0]),
                    "center": (cx, cy)
                })
                
        return tracked_objects