import math

class TrajectoryManager:
    def __init__(self, fps):
        self.fps = fps
        self.history = {}

    def update(self, tracked_objects):
        for obj in tracked_objects:
            track_id = obj["id"]
            center = obj["center"]
            
            if track_id not in self.history:
                self.history[track_id] = {
                    "class": obj.get("class", "unknown"), 
                    "trajectory": [],
                    "speed": 0,
                    "acceleration": 0,
                    "heading": 0,
                    "previous_speed": 0
                }

            vehicle = self.history[track_id]
            vehicle["trajectory"].append(center)
            
            if len(vehicle["trajectory"]) > 50:
                vehicle["trajectory"].pop(0)

            self.calculate_behavior(track_id)

        return self.history

    def calculate_behavior(self, track_id):
        vehicle = self.history[track_id]
        trajectory = vehicle["trajectory"]

        if len(trajectory) < 2:
            return

        x1, y1 = trajectory[-2]
        x2, y2 = trajectory[-1]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        speed = distance * self.fps
        acceleration = (speed - vehicle["previous_speed"]) * self.fps
        heading = math.degrees(math.atan2(y2 - y1, x2 - x1))

        vehicle["speed"] = speed
        vehicle["acceleration"] = acceleration
        vehicle["heading"] = heading
        vehicle["previous_speed"] = speed

    def get_history(self):
        return self.history