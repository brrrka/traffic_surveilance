"""Per-camera metric calibration for fixed CCTV views.

A planar homography maps image footpoints to a measured road plane. Absolute
metric scale requires physical ground correspondences; view labels, vanishing
points, and vehicle dimensions alone do not establish that scale.
"""
from __future__ import annotations

import json
import math
from typing import Any

import cv2
import numpy as np

VIEWS = ("auto", "side", "front", "rear", "diagonal", "elevated", "top")


def _points(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 4:
        raise ValueError(f"{name} harus berisi minimal 4 pasang koordinat (x, y).")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} mengandung nilai non-finite.")
    if len(np.unique(arr, axis=0)) < 4:
        raise ValueError(f"{name} harus memiliki minimal 4 titik berbeda.")
    if np.linalg.matrix_rank(np.column_stack([arr, np.ones(len(arr))])) < 3:
        raise ValueError(f"{name} segaris; homography bidang 2D tidak dapat dihitung.")
    return arr


class CameraCalibration:
    """One calibration instance belongs to exactly one fixed CCTV camera."""

    def __init__(self, camera_view="auto", profile=None, frame_size=None, allow_side_prior=True):
        self.camera_view = str(camera_view or "auto").lower()
        if self.camera_view not in VIEWS:
            raise ValueError(f"camera_view harus salah satu dari {VIEWS}.")
        if isinstance(profile, str) and profile.strip():
            profile = json.loads(profile)
        self.profile = profile or {}
        self.frame_size = tuple(map(int, frame_size)) if frame_size else None  # (width, height)
        self.allow_side_prior = bool(allow_side_prior)
        self.H = None
        self.status = "uncalibrated"
        self.method = "image_motion_only"
        self.quality = "Tidak terkalibrasi"
        self.warnings = []
        self.reprojection_rmse_m = None
        self.inlier_ratio = None
        self.camera_id = self.profile.get("camera_id") if isinstance(self.profile, dict) else None
        self.camera_name = self.profile.get("camera_name") if isinstance(self.profile, dict) else None
        self.profile_version = self.profile.get("profile_version") if isinstance(self.profile, dict) else None
        self._build()

    def _build(self):
        if not isinstance(self.profile, dict) or not self.profile:
            self.warnings.append(
                "Profil metrik kamera belum tersedia. Sistem tetap menghitung px/s untuk semua kendaraan. "
                "AUTO hanya boleh memberi estimasi km/jam berkeyakinan rendah bila prior geometri kendaraan "
                "terlihat kuat; tanpa anchor meter, nilai tersebut bukan kalibrasi metrik."
            )
            return

        calibration = self.profile.get("calibration", self.profile)
        if not isinstance(calibration, dict):
            raise ValueError("Bagian calibration pada profil kamera harus berupa object JSON.")
        mode = calibration.get("mode", "pixel_only")
        if mode in ("pixel_only", "uncalibrated", None):
            self.warnings.append("Profil kamera tersimpan, tetapi belum memiliki titik referensi meter.")
            return
        if mode != "ground_plane_points":
            raise ValueError("Mode kalibrasi yang didukung adalah ground_plane_points atau pixel_only.")

        image_size = calibration.get("image_size") or self.profile.get("image_size")
        if image_size:
            src_w = int(image_size.get("width", 0))
            src_h = int(image_size.get("height", 0))
            if src_w <= 0 or src_h <= 0:
                raise ValueError("image_size harus memiliki width dan height positif.")
        else:
            src_w = src_h = 0

        target_w, target_h = self.frame_size or (src_w, src_h)

        if "image_points_normalized" in calibration:
            if target_w <= 0 or target_h <= 0:
                raise ValueError("Resolusi frame diperlukan untuk image_points_normalized.")
            norm = _points(calibration["image_points_normalized"], "image_points_normalized")
            if np.any(norm < 0) or np.any(norm > 1):
                raise ValueError("Koordinat image_points_normalized harus berada pada rentang 0 sampai 1.")
            if src_w and src_h:
                src_ratio = src_w / src_h
                dst_ratio = target_w / target_h
                if abs(src_ratio - dst_ratio) / src_ratio > 0.01:
                    raise ValueError(
                        "Rasio aspek video berbeda lebih dari 1% dari gambar kalibrasi. "
                        "Ambil ulang frame kalibrasi dari stream/resolusi CCTV yang sama."
                    )
            image_points = norm * np.array([target_w - 1, target_h - 1], dtype=np.float64)
        else:
            image_points = _points(calibration.get("image_points", []), "image_points")
            if src_w and src_h and target_w > 0 and target_h > 0 and (src_w != target_w or src_h != target_h):
                src_ratio = src_w / src_h
                dst_ratio = target_w / target_h
                if abs(src_ratio - dst_ratio) / src_ratio > 0.01:
                    raise ValueError("Rasio aspek video tidak cocok dengan gambar kalibrasi.")
                image_points = image_points * np.array([target_w / src_w, target_h / src_h])

        world_points = _points(calibration.get("world_points_m", []), "world_points_m")
        if len(image_points) != len(world_points):
            raise ValueError("Jumlah image_points dan world_points_m harus sama.")

        # Four correspondences define a homography exactly; >=5 points enable
        # outlier rejection and a more meaningful residual diagnostic.
        method = cv2.RANSAC if len(image_points) >= 5 else 0
        threshold = float(calibration.get("ransac_threshold_m", 0.35))
        H, mask = cv2.findHomography(image_points, world_points, method, threshold)
        if H is None or not np.isfinite(H).all() or abs(float(np.linalg.det(H))) < 1e-15:
            raise ValueError("Homography gagal dihitung. Periksa titik gambar dan koordinat meter.")

        inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(image_points), dtype=bool)
        if int(inliers.sum()) < 4:
            raise ValueError("Kurang dari 4 titik kalibrasi yang konsisten; periksa pasangan titik.")
        projected = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2).astype(np.float64), H
        ).reshape(-1, 2)
        residual = np.linalg.norm(projected - world_points, axis=1)
        rmse = float(np.sqrt(np.mean(np.square(residual[inliers]))))
        if not math.isfinite(rmse) or rmse > 0.75:
            raise ValueError(f"Reprojection RMSE {rmse:.3f} m terlalu besar (>0,75 m). Periksa titik referensi.")

        self.H = H
        self.status = "calibrated_reference"
        self.method = "ground_plane_homography"
        self.quality = "Referensi bidang jalan tersimpan"
        self.reprojection_rmse_m = round(rmse, 4)
        self.inlier_ratio = round(float(inliers.mean()), 4)
        if len(image_points) == 4:
            self.warnings.append(
                "Kalibrasi memakai tepat 4 titik: transformasi dapat dihitung, tetapi tidak ada redundansi "
                "untuk menguji outlier. Disarankan 6–8 titik yang tersebar di area jalan."
            )
        self.warnings.append(
            "Homography berlaku pada bidang jalan yang diasumsikan planar dan untuk CCTV tetap. "
            "Footpoint kendaraan harus berada pada bidang jalan yang sama."
        )
        self.warnings.append(
            "Profil ini khusus untuk kamera ini; jangan digunakan untuk CCTV lain atau kamera yang berubah zoom/pan."
        )

    def ground_point(self, point_xy):
        if self.H is None or point_xy is None:
            return None
        point = np.asarray(point_xy, dtype=np.float64).reshape(1, 1, 2)
        mapped = cv2.perspectiveTransform(point, self.H).reshape(2)
        if not np.isfinite(mapped).all():
            return None
        return float(mapped[0]), float(mapped[1])

    def speed(self, velocity_px_s, bbox_xyxy=None, displacement_px=None, dt_s=None,
              location_px=None, previous_location_px=None):
        """Compatibility helper; metric speed is only returned for a valid H."""
        base = {
            "speed_kmh": None, "velocity_m_s": None, "status": "uncalibrated",
            "quality": "Tidak terkalibrasi", "method": "image_motion_only",
            "warning": "Tidak ada referensi skala meter yang valid.",
        }
        if self.H is not None and location_px is not None and previous_location_px is not None and dt_s and dt_s > 0:
            p1, p2 = self.ground_point(previous_location_px), self.ground_point(location_px)
            if p1 is not None and p2 is not None:
                vx, vy = (p2[0] - p1[0]) / dt_s, (p2[1] - p1[1]) / dt_s
                speed = math.hypot(vx, vy)
                if math.isfinite(speed) and speed <= 80:
                    return dict(base, speed_kmh=round(speed * 3.6, 2),
                                velocity_m_s=(vx, vy), status="calibrated_reference",
                                quality=self.quality, method=self.method,
                                warning="Bergantung pada akurasi titik referensi, homography, dan tracking.")
        # Legacy fallback: only a near-horizontal side profile, explicitly labelled
        # as a weak vehicle-dimension prior. Never used for other views.
        view = self.camera_view
        if self.allow_side_prior and view == "side" and bbox_xyxy is not None and velocity_px_s:
            x1, y1, x2, y2 = map(float, bbox_xyxy)
            w, h = x2 - x1, y2 - y1
            vx, vy = map(float, velocity_px_s)
            px_s = math.hypot(vx, vy)
            if w >= 45 and h >= 22 and 2.0 <= w / max(h, 1) <= 4.6 and abs(vx) >= abs(vy) * 2.5:
                speed = px_s * (4.4 / w) * 3.6
                if 0 <= speed <= 180:
                    return dict(base, speed_kmh=round(speed, 2), status="estimated_prior",
                                quality="Rendah / prior dimensi generik", method="side_vehicle_length_prior",
                                warning="ESTIMASI KASAR: panjang mobil diasumsikan 4,4 m; bukan kalibrasi metrik kamera.")
        return base

    def describe(self):
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "profile_version": self.profile_version,
            "camera_view": self.camera_view,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "status": self.status,
            "method": self.method,
            "quality": self.quality,
            "reprojection_rmse_m": self.reprojection_rmse_m,
            "inlier_ratio": self.inlier_ratio,
            "warnings": list(self.warnings),
            "estimated_prior_supported": self.allow_side_prior and self.camera_view in ("side", "auto"),
            "metric_scale_available": self.H is not None,
            "absolute_speed_note": (
                "Homography bidang jalan tersedia." if self.H is not None else
                "Skala meter absolut tidak teridentifikasi; AUTO dapat memberi prior kendaraan berkeyakinan rendah atau hanya px/s."
            ),
        }
