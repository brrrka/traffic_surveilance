"""ANCHOR VI independent vehicle-speed engine.

Absolute km/h needs a metric anchor. This engine always reports image speed
(px/s), uses a valid per-camera ground-plane homography when supplied, and
allows only a clearly labelled low-confidence vehicle-length prior when the
vehicle presents a side-like projected length aligned with its image motion.
It never silently converts arbitrary pixel motion to km/h.
"""
from __future__ import annotations

import math
from statistics import median


BUILD_ID = "ANCHOR-VI-speed-engine-1.1-impact-speed"
GENERIC_CAR_LENGTH_M = 4.4


def _angle_diff_180(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _bbox_features(history):
    boxes = history.get("bbox_history") or []
    if not boxes:
        return None
    rows = []
    for box in boxes[-12:]:
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = map(float, box)
        w, h = x2 - x1, y2 - y1
        if w > 2 and h > 2:
            rows.append((w, h, w / h))
    if not rows:
        return None
    return {
        "median_width_px": median(r[0] for r in rows),
        "median_height_px": median(r[1] for r in rows),
        "median_aspect_ratio": median(r[2] for r in rows),
    }


def _side_like_length_prior(history, camera_view="auto"):
    """Estimate a low-confidence metric speed from a visible vehicle axis.

    This is a dimension prior, NOT camera calibration. It is only attempted
    when the vehicle's major image-box axis is reasonably aligned with motion.
    The returned vector is in an approximate image-aligned metric coordinate
    system and must never be described as calibrated ground-plane velocity.
    """
    features = _bbox_features(history)
    if not features:
        return None

    view = str(camera_view or "auto").lower()
    if view not in ("auto", "side", "diagonal"):
        return None

    vx = float(history.get("velocity_x") or 0.0)
    vy = float(history.get("velocity_y") or 0.0)
    px_s = float(history.get("speed") or 0.0)
    observations = int(history.get("observations") or 0)
    if px_s < 1.0 or observations < 4:
        return None

    width = float(features["median_width_px"])
    height = float(features["median_height_px"])
    ratio = max(width, height) / max(min(width, height), 1.0)

    # Major axis of the axis-aligned vehicle box.
    major_axis_deg = 0.0 if width >= height else 90.0
    motion_axis_deg = math.degrees(math.atan2(vy, vx)) % 180.0
    alignment = _angle_diff_180(major_axis_deg, motion_axis_deg)

    # Relaxed for oblique CCTV views, but still requires a visible elongated
    # vehicle and motion reasonably aligned with its projected long axis.
    if ratio < 1.20 or ratio > 6.0 or alignment > 55.0:
        return None

    apparent_length_px = max(width, height)
    if apparent_length_px < 18.0:
        return None

    scale_m_per_px = GENERIC_CAR_LENGTH_M / apparent_length_px
    speed_kmh = px_s * scale_m_per_px * 3.6
    if not math.isfinite(speed_kmh) or speed_kmh < 0.0 or speed_kmh > 160.0:
        return None

    # Confidence remains deliberately low because 4.4 m is a generic prior
    # and perspective/foreshortening cannot be solved from one monocular view.
    shape_score = min(1.0, max(0.0, (ratio - 1.20) / 1.80))
    alignment_score = max(0.0, 1.0 - alignment / 60.0)
    size_score = min(1.0, apparent_length_px / 60.0)
    confidence = round(min(0.35, 0.12 + 0.10 * shape_score
                           + 0.08 * alignment_score + 0.05 * size_score), 2)

    velocity_m_s = (vx * scale_m_per_px, vy * scale_m_per_px)
    return {
        "speed_kmh": round(speed_kmh, 2),
        "velocity_m_s": (round(velocity_m_s[0], 4), round(velocity_m_s[1], 4)),
        "scale_m_per_px": round(scale_m_per_px, 6),
        "status": "estimated_prior",
        "method": "projected_vehicle_length_prior_low_confidence",
        "confidence": confidence,
        "confidence_label": "Rendah",
        "apparent_vehicle_length_px": round(apparent_length_px, 2),
        "assumed_vehicle_length_m": GENERIC_CAR_LENGTH_M,
        "axis_alignment_deg": round(alignment, 1),
        "warning": (
            "ESTIMASI RENDAH: panjang kendaraan diasumsikan 4,4 m dari sumbu "
            "bounding box yang tampak. Ini bukan kalibrasi kamera atau pengukuran "
            "ground-plane; validasi dengan radar/speedometer sebelum dipakai "
            "sebagai data kuantitatif."
        ),
    }


class VehicleSpeedEngine:
    """Maintain independent speed histories for every tracked vehicle."""

    def __init__(self, fps: float, calibration=None, camera_view: str = "auto", mode: str = "both"):
        self.fps = float(fps)
        self.calibration = calibration
        self.camera_view = str(camera_view or "auto").lower()
        self.mode = mode
        self.samples = {}

    def update(self, histories, tracked_objects, frame_index: int):
        for obj in tracked_objects or []:
            try:
                tid = int(obj["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if obj.get("class") != "car" or tid not in histories:
                continue

            h = histories[tid]
            px_s = h.get("speed")
            if px_s is None:
                continue
            try:
                px_s = float(px_s)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(px_s):
                continue

            speed_kmh = h.get("speed_kmh")
            status = h.get("speed_status", "uncalibrated")
            method = h.get("speed_method", "image_motion_only")
            vx = float(h.get("velocity_x") or 0.0)
            vy = float(h.get("velocity_y") or 0.0)
            velocity_m_s = h.get("velocity_m_s")
            scale_m_per_px = None
            confidence = 0.0
            confidence_label = "Tidak tersedia"
            warning = h.get("speed_warning") or "Tidak ada skala meter; hanya kecepatan citra (px/s)."
            apparent_length_px = None

            if status == "calibrated_reference" and speed_kmh is not None:
                confidence = 0.65
                confidence_label = "Sedang"
                method = "per_camera_ground_plane_homography"
                warning = (
                    "Kecepatan metrik memakai homography bidang jalan pada kamera ini. "
                    "Akurasi tetap bergantung pada kualitas titik referensi, footpoint, dan tracking."
                )
            elif self.calibration is not None and getattr(self.calibration, "H", None) is None:
                prior = _side_like_length_prior(h, self.camera_view)
                if prior:
                    speed_kmh = prior["speed_kmh"]
                    velocity_m_s = prior["velocity_m_s"]
                    scale_m_per_px = prior["scale_m_per_px"]
                    status = prior["status"]
                    method = prior["method"]
                    confidence = prior["confidence"]
                    confidence_label = prior["confidence_label"]
                    warning = prior["warning"]
                    apparent_length_px = prior["apparent_vehicle_length_px"]
                elif status == "estimated_prior" and speed_kmh is not None:
                    # Preserve a legacy calibration prior if it was already
                    # generated by CameraCalibration.speed().
                    speed_m_s = float(speed_kmh) / 3.6
                    norm_px = math.hypot(vx, vy)
                    if norm_px > 1e-6:
                        scale_m_per_px = speed_m_s / norm_px
                        velocity_m_s = (vx * scale_m_per_px, vy * scale_m_per_px)
                    confidence = 0.18
                    confidence_label = "Rendah"
                    warning = warning or "Estimasi prior dimensi kendaraan; bukan kalibrasi metrik."
                else:
                    speed_kmh = None
                    velocity_m_s = None
                    status = "uncalibrated"
                    method = "image_motion_only"
                    confidence = 0.0
                    confidence_label = "Tidak tersedia"
                    warning = (
                        "AUTO belum menemukan sumbu kendaraan yang cukup kuat untuk prior dimensi. "
                        "Kecepatan metrik tidak dihitung; px/s tetap tersedia."
                    )

            if speed_kmh is not None:
                try:
                    speed_kmh = round(float(speed_kmh), 2)
                except (TypeError, ValueError):
                    speed_kmh = None

            h["speed_kmh"] = speed_kmh
            h["speed_status"] = status
            h["speed_method"] = method
            h["speed_confidence"] = confidence
            h["speed_confidence_label"] = confidence_label
            h["speed_warning"] = warning
            h["velocity_m_s"] = velocity_m_s
            h["estimated_scale_m_per_px"] = scale_m_per_px

            # Keep the trajectory's last speed-history row in sync. The
            # matrix builder reads these rows immediately before the event.
            speed_history = h.get("speed_history") or []
            if speed_history and int(speed_history[-1].get("frame", -1)) == int(frame_index):
                speed_history[-1].update({
                    "speed_kmh": speed_kmh,
                    "speed_status": status,
                    "speed_method": method,
                    "speed_confidence": confidence,
                    "speed_confidence_label": confidence_label,
                    "speed_warning": warning,
                    "apparent_vehicle_length_px": apparent_length_px,
                    "estimated_scale_m_per_px": scale_m_per_px,
                    "velocity_m_s": velocity_m_s,
                    "world_velocity_m_s": (
                        velocity_m_s if status == "calibrated_reference" else None
                    ),
                })

            # Coordinates of this exact frame accompany the speed sample.
            # Required for a time-synchronized pre-contact closing-speed estimate
            # even if bounded TrajectoryManager history later evicts this frame.
            footpoint_now = (h.get("footpoint_trajectory") or [None])[-1]
            world_point_now = (h.get("world_trajectory") or [None])[-1]
            sample = {
                "footpoint_px": footpoint_now,
                "world_point_m": world_point_now,
                "frame": int(frame_index),
                "time_s": round((int(frame_index) - 1) / self.fps, 3),
                "speed_px_s": round(px_s, 2),
                "speed_kmh": speed_kmh,
                "speed_status": status,
                "speed_method": method,
                "confidence": confidence,
                "confidence_label": confidence_label,
                "warning": warning,
                "velocity_m_s": velocity_m_s,
                "estimated_scale_m_per_px": scale_m_per_px,
            }
            self.samples.setdefault(tid, []).append(sample)

    def summaries(self):
        output = []
        for tid, samples in sorted(self.samples.items()):
            px = [float(s["speed_px_s"]) for s in samples if s.get("speed_px_s") is not None]
            metric = [float(s["speed_kmh"]) for s in samples
                      if s.get("speed_kmh") is not None and s.get("speed_status") in ("calibrated_reference", "estimated_prior")]
            statuses = {s.get("speed_status") for s in samples}
            if "calibrated_reference" in statuses:
                status = "calibrated_reference"
            elif "estimated_prior" in statuses:
                status = "estimated_prior"
            else:
                status = "uncalibrated"
            last = samples[-1] if samples else {}
            output.append({
                "vehicle_id": int(tid),
                "observations_with_speed": len(samples),
                # Mean across every valid tracked frame (not the last 240 frames).
                "average_speed_px_s": round(sum(px)/len(px), 2) if len(px) >= 2 else None,
                "average_speed_kmh": round(sum(metric)/len(metric), 2) if len(metric) >= 2 and len({x.get('speed_status') for x in samples if x.get('speed_kmh') is not None and x.get('speed_status') in ('calibrated_reference', 'estimated_prior')}) == 1 else None,
                # Legacy keys retained so older dashboard versions keep working.
                "median_speed_px_s": round(median(px), 2) if px else None,
                "median_speed_kmh": round(median(metric), 2) if metric else None,
                "speed_status": status,
                "speed_method": last.get("speed_method", "image_motion_only"),
                "confidence": last.get("confidence", 0.0),
                "confidence_label": last.get("confidence_label", "Tidak tersedia"),
                "warning": last.get("warning", "Tidak ada data kecepatan."),
            })
        return output

    def profiles(self, max_points_per_vehicle: int = 180):
        output = []
        for tid, samples in sorted(self.samples.items()):
            if len(samples) > max_points_per_vehicle:
                step = math.ceil(len(samples) / max_points_per_vehicle)
                points = samples[::step]
                if points[-1] is not samples[-1]:
                    points.append(samples[-1])
            else:
                points = samples
            output.append({"vehicle_id": int(tid), "points": points})
        return output


def infer_camera_view(histories):
    """A weak display hint only; it never creates metric calibration."""
    aspect_values = []
    horizontal_dominance = []
    eligible = 0
    for h in (histories or {}).values():
        feat = _bbox_features(h)
        if not feat or int(h.get("observations", 0)) < 5:
            continue
        vx = abs(float(h.get("velocity_x") or 0.0))
        vy = abs(float(h.get("velocity_y") or 0.0))
        if vx + vy < 1:
            continue
        eligible += 1
        aspect_values.append(feat["median_aspect_ratio"])
        horizontal_dominance.append(max(vx, vy) / max(min(vx, vy), 1.0))
    if not eligible:
        return {"label": "Belum dapat diperkirakan", "confidence": "Tidak tersedia", "is_metric_calibration": False}
    side_fraction = sum(1 for a, d in zip(aspect_values, horizontal_dominance) if a >= 2.0 and d >= 1.5) / eligible
    if side_fraction >= 0.6:
        label, confidence = "Side-like (indikasi tampak samping)", "Rendah"
    elif median(aspect_values) <= 1.35:
        label, confidence = "Front/rear-like (indikasi depan/belakang)", "Rendah"
    else:
        label, confidence = "Diagonal/elevated atau oblique (belum dapat dibedakan pasti)", "Rendah"
    return {
        "label": label,
        "confidence": confidence,
        "tracks_used": eligible,
        "median_vehicle_box_aspect": round(median(aspect_values), 3),
        "is_metric_calibration": False,
        "warning": "Petunjuk perspektif otomatis berbasis bentuk bounding box dan arah gerak; bukan estimasi pose kamera terkalibrasi.",
    }
