import math

# Kelas untuk mengelola dan menganalisis trajektori kendaraan
class TrajectoryManager:

    def __init__(self, fps):

        self.fps = fps

        self.history = {}

    # Fungsi untuk memperbarui trajektori kendaraan dengan data pelacakan baru dan menghitung perilaku kendaraan
    def update(self, tracked_objects):

        # Iterasi melalui setiap objek yang dilacak dan perbarui trajektori serta perilaku kendaraan
        for obj in tracked_objects:

            track_id = obj["id"]

            center = obj["center"]

            # Jika track_id belum ada dalam history, inisialisasi data untuk kendaraan baru
            if track_id not in self.history:

                self.history[track_id] = {

                    "trajectory": [],

                    "speed": 0,

                    "acceleration": 0,

                    "heading": 0,

                    "previous_speed": 0

                }

            vehicle = self.history[track_id]

            vehicle["trajectory"].append(center)

            # Simpan maksimal 50 titik
            if len(vehicle["trajectory"]) > 50:

                vehicle["trajectory"].pop(0)

            self.calculate_behavior(track_id)

        return self.history

    # Fungsi untuk menghitung kecepatan, percepatan, dan arah kendaraan berdasarkan trajektori yang tersimpan
    def calculate_behavior(self, track_id):

        vehicle = self.history[track_id]

        trajectory = vehicle["trajectory"]

        # Perlu minimal 2 titik untuk menghitung kecepatan dan arah
        if len(trajectory) < 2:

            return

        x1, y1 = trajectory[-2]

        x2, y2 = trajectory[-1]

        distance = math.sqrt(

            (x2 - x1)**2 +

            (y2 - y1)**2

        )

        # Kecepatan dalam piksel per detik
        speed = distance * self.fps

        # Percepatan dalam piksel per detik kuadrat
        acceleration = (

            speed -

            vehicle["previous_speed"]

        ) * self.fps

        # Arah dalam derajat (0-360)
        heading = math.degrees(

            math.atan2(

                y2 - y1,

                x2 - x1

            )

        )

        vehicle["speed"] = speed

        vehicle["acceleration"] = acceleration

        vehicle["heading"] = heading

        vehicle["previous_speed"] = speed

    def get_history(self):

        return self.history