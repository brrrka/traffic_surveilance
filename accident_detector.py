import math

class AccidentDetector:
    def __init__(self):
        self.decel_threshold = -5000
        self.distance_threshold = 100

    def detect(self, histories):
        accidents = []
        vehicle_ids = list(histories.keys())

        for i in range(len(vehicle_ids)):
            id1 = vehicle_ids[i]
            data1 = histories[id1]

            if data1["acceleration"] < self.decel_threshold:
                if not data1["trajectory"]:
                    continue
                    
                center1 = data1["trajectory"][-1]

                for j in range(i + 1, len(vehicle_ids)):
                    id2 = vehicle_ids[j]
                    data2 = histories[id2]
                    
                    if not data2["trajectory"]:
                        continue
                        
                    center2 = data2["trajectory"][-1]
                    dist = math.hypot(center1[0] - center2[0], center1[1] - center2[1])

                    if dist < self.distance_threshold:
                        accidents.append({
                            "v1": id1,
                            "v2": id2,
                            "location": center1
                        })

        return accidents