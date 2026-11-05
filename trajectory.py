"""Vehicle trajectory, image motion, and per-camera ground-plane speed."""
import math
import numpy as np


class TrajectoryManager:
    def __init__(self, fps, calibration=None, max_points=240, motion_window=11):
        self.fps = float(fps)
        if not math.isfinite(self.fps) or not 0 < self.fps <= 240:
            raise ValueError("FPS tidak valid")
        self.calibration = calibration
        self.max_points = max(40, int(max_points))
        self.motion_window = max(5, int(motion_window))
        self.frame_index = 0
        self.history = {}

    @staticmethod
    def _regression(frames, points, fps):
        """Least-squares x/y velocity in coordinate units per second."""
        f = np.asarray(frames, dtype=np.float64)
        p = np.asarray(points, dtype=np.float64)
        if len(f) < 2 or p.shape != (len(f), 2):
            return 0.0, 0.0
        t = (f - f[0]) / float(fps)
        if t[-1] - t[0] < 1e-6:
            return 0.0, 0.0
        design = np.column_stack([t, np.ones_like(t)])
        cx, *_ = np.linalg.lstsq(design, p[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(design, p[:, 1], rcond=None)
        return float(cx[0]), float(cy[0])

    def _new_history(self, tid):
        return {
            "track_id": tid, "class": "car",
            "trajectory": [], "footpoint_trajectory": [], "analysis_trajectory": [],
            "world_trajectory": [], "bbox_history": [], "frame_indices": [],
            "speed_history": [], "speed": None, "velocity_x": 0.0, "velocity_y": 0.0,
            "world_velocity_x": None, "world_velocity_y": None,
            "acceleration": None, "acceleration_m_s2": None,
            "heading": None, "analysis_heading": None,
            "speed_kmh": None, "speed_status": "uncalibrated",
            "speed_method": "image_motion_only", "speed_warning": "Belum terkalibrasi",
            "velocity_m_s": None, "analysis_coordinate_space": "image_px",
            "active": False, "last_seen_frame": None, "observations": 0,
        }

    def update(self, tracked_objects):
        self.frame_index += 1
        current = set()
        for obj in tracked_objects or []:
            if obj.get("class") != "car":
                continue
            tid = int(obj["id"])
            if tid in current:
                continue
            current.add(tid)
            bbox = tuple(float(x) for x in obj["bbox"])
            center = tuple(float(x) for x in obj.get("center", ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)))
            footpoint = tuple(float(x) for x in obj.get("footpoint", ((bbox[0] + bbox[2]) / 2, bbox[3])))

            if tid not in self.history:
                self.history[tid] = self._new_history(tid)
            h = self.history[tid]
            h["trajectory"].append(center)
            h["footpoint_trajectory"].append(footpoint)
            h["bbox_history"].append(bbox)
            h["frame_indices"].append(self.frame_index)

            world_point = self.calibration.ground_point(footpoint) if self.calibration and self.calibration.H is not None else None
            h["world_trajectory"].append(world_point)
            if world_point is not None:
                h["analysis_trajectory"].append(world_point)
                h["analysis_coordinate_space"] = "ground_plane_m"
            else:
                h["analysis_trajectory"].append(footpoint)
                h["analysis_coordinate_space"] = "image_px"

            for key in ("trajectory", "footpoint_trajectory", "analysis_trajectory", "world_trajectory", "bbox_history", "frame_indices"):
                if len(h[key]) > self.max_points:
                    h[key].pop(0)
            h["observations"] += 1
            h["active"] = True
            h["last_seen_frame"] = self.frame_index
            self.calculate_behavior(tid)

        for tid, history in self.history.items():
            if tid not in current:
                history["active"] = False
        return self.history

    def calculate_behavior(self, tid):
        h = self.history[tid]
        frames = h["frame_indices"]
        if len(frames) < 2:
            return
        if frames[-1] - frames[-2] > 5:
            h.update({"speed": None, "speed_kmh": None, "velocity_m_s": None,
                      "heading": None, "analysis_heading": None,
                      "speed_status": "uncalibrated", "speed_method": "tracking_gap",
                      "speed_warning": "Tracking terputus; kecepatan tidak dihitung melintasi gap."})
            return

        # Jangan gabungkan regresi sebelum/sesudah gap pelacakan >5 frame.
        # Riwayat lengkap tetap dipertahankan untuk classifier dan audit.
        segment_start = max(0, len(frames) - self.motion_window)
        for i in range(len(frames) - 1, segment_start, -1):
            if frames[i] - frames[i - 1] > 5:
                segment_start = i
                break
        sample_frames = frames[segment_start:]
        sample_pixels = h["footpoint_trajectory"][segment_start:]
        n = len(sample_frames)
        vx_px, vy_px = self._regression(sample_frames, sample_pixels, self.fps)
        px_s = math.hypot(vx_px, vy_px)
        heading_px = math.degrees(math.atan2(vy_px, vx_px)) % 360.0 if px_s >= 1.0 else None

        speed_kmh = None
        velocity_m_s = None
        speed_status = "uncalibrated"
        speed_method = "image_motion_only"
        speed_warning = "Tidak ada referensi meter untuk kamera ini; hanya px/s yang tersedia."
        world_vx = world_vy = None
        acceleration_m_s2 = None
        analysis_heading = heading_px

        if self.calibration and self.calibration.H is not None:
            world_frames = []
            world_points = []
            for f, point in zip(sample_frames, h["world_trajectory"][-n:]):
                if point is not None:
                    world_frames.append(f)
                    world_points.append(point)
            if len(world_points) >= 5 and world_frames[-1] - world_frames[0] >= 3:
                world_vx, world_vy = self._regression(world_frames, world_points, self.fps)
                world_speed = math.hypot(world_vx, world_vy)
                # Guard only obvious numerical/tracking failures; the threshold
                # is a safety check, not a physical speed-classification model.
                if math.isfinite(world_speed) and world_speed <= 80:
                    speed_kmh = round(world_speed * 3.6, 2)
                    velocity_m_s = (world_vx, world_vy)
                    speed_status = "calibrated_reference"
                    speed_method = "per_camera_ground_plane_homography"
                    speed_warning = "Kecepatan dihitung pada bidang jalan kamera ini; ketelitian bergantung pada titik kalibrasi dan footpoint."
                    analysis_heading = math.degrees(math.atan2(world_vy, world_vx)) % 360.0 if world_speed >= 0.15 else None
                    h["analysis_coordinate_space"] = "ground_plane_m"

                    previous_metric = h["speed_history"][-1] if h["speed_history"] else None
                    if previous_metric and previous_metric.get("world_speed_m_s") is not None:
                        elapsed = (frames[-1] - previous_metric["frame"]) / self.fps
                        if elapsed > 0:
                            acceleration_m_s2 = (world_speed - previous_metric["world_speed_m_s"]) / elapsed
                else:
                    speed_warning = "Hasil homography menghasilkan kecepatan di luar batas pemeriksaan; nilai km/jam disembunyikan."
            else:
                speed_warning = "Belum cukup titik ground-plane yang valid untuk menghitung kecepatan metrik."
        elif self.calibration:
            # Side-view prior remains explicitly estimated, never labelled calibrated.
            result = self.calibration.speed((vx_px, vy_px), h["bbox_history"][-1],
                                            (vx_px, vy_px), 1.0 / self.fps)
            speed_kmh = result.get("speed_kmh")
            speed_status = result.get("status", "uncalibrated")
            speed_method = result.get("method", "image_motion_only")
            speed_warning = result.get("warning", speed_warning)

        previous = h["speed_history"][-1] if h["speed_history"] else None
        acceleration_px = None
        if previous and previous.get("speed_px_s") is not None:
            elapsed = (frames[-1] - previous["frame"]) / self.fps
            if elapsed > 0:
                acceleration_px = (px_s - previous["speed_px_s"]) / elapsed

        h.update({
            "speed": px_s, "velocity_x": vx_px, "velocity_y": vy_px,
            "world_velocity_x": world_vx, "world_velocity_y": world_vy,
            "acceleration": acceleration_px, "acceleration_m_s2": acceleration_m_s2,
            "heading": heading_px, "analysis_heading": analysis_heading,
            "speed_kmh": speed_kmh, "speed_status": speed_status,
            "speed_method": speed_method, "speed_warning": speed_warning,
            "velocity_m_s": velocity_m_s,
        })
        h["speed_history"].append({
            "frame": frames[-1], "speed_px_s": round(px_s, 3),
            "heading_deg": heading_px, "vx_px_s": vx_px, "vy_px_s": vy_px,
            "speed_kmh": speed_kmh, "speed_status": speed_status,
            "velocity_m_s": velocity_m_s,
            "world_speed_m_s": math.hypot(world_vx, world_vy) if world_vx is not None and world_vy is not None else None,
            "world_velocity_m_s": velocity_m_s,
            "coordinate_space": h["analysis_coordinate_space"],
        })
        if len(h["speed_history"]) > self.max_points:
            h["speed_history"].pop(0)

    def get_active_history(self):
        return {key: value for key, value in self.history.items() if value.get("active")}

    def get_history(self):
        return self.history
