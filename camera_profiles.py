"""Persistent per-CCTV calibration profile storage."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from pathlib import Path

from camera_calibration import CameraCalibration, VIEWS

ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "data" / "camera_profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_camera_id(camera_id: str) -> str:
    camera_id = str(camera_id or "").strip()
    if not CAMERA_ID_RE.fullmatch(camera_id):
        raise ValueError("ID kamera hanya boleh berisi huruf, angka, underscore, dan tanda hubung (1–64 karakter).")
    return camera_id


def _path(camera_id: str) -> Path:
    return PROFILE_DIR / f"{validate_camera_id(camera_id)}.json"


def list_camera_profiles():
    result = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            cal = profile.get("calibration", {})
            result.append({
                "camera_id": profile.get("camera_id", path.stem),
                "camera_name": profile.get("camera_name", path.stem),
                "camera_view": profile.get("camera_view", "auto"),
                "calibration_status": "calibrated_reference" if cal.get("mode") == "ground_plane_points" else "uncalibrated",
                "point_count": len(cal.get("world_points_m", [])),
                "profile_version": profile.get("profile_version", 1),
                "updated_at": profile.get("updated_at"),
            })
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return result


def get_camera_profile(camera_id: str):
    path = _path(camera_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_camera_profile(camera_id: str, payload: dict):
    camera_id = validate_camera_id(camera_id)
    if not isinstance(payload, dict):
        raise ValueError("Payload profil kamera harus berupa object JSON.")
    camera_name = str(payload.get("camera_name") or camera_id).strip()
    if not camera_name:
        raise ValueError("Nama kamera tidak boleh kosong.")
    camera_view = str(payload.get("camera_view") or "auto").lower()
    if camera_view not in VIEWS:
        raise ValueError(f"camera_view harus salah satu dari {VIEWS}.")

    image_size = payload.get("image_size") or {}
    calibration_payload = payload.get("calibration") or {}
    if not isinstance(calibration_payload, dict):
        raise ValueError("calibration harus berupa object JSON.")

    has_points = bool(calibration_payload.get("image_points_normalized") or calibration_payload.get("image_points"))
    if has_points:
        calibration_payload = dict(calibration_payload)
        calibration_payload["mode"] = "ground_plane_points"
        calibration_payload["image_size"] = image_size or calibration_payload.get("image_size")
        if not calibration_payload.get("image_size"):
            raise ValueError("Resolusi gambar kalibrasi (image_size) wajib diisi.")
    else:
        calibration_payload = {"mode": "pixel_only", "image_size": image_size or None}

    profile = {
        "schema_version": 2,
        "profile_version": 1,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "camera_view": camera_view,
        "image_size": image_size or None,
        "calibration": calibration_payload,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    existing = get_camera_profile(camera_id)
    if existing:
        profile["profile_version"] = int(existing.get("profile_version", 1)) + 1
        if not has_points and existing.get("calibration", {}).get("mode") == "ground_plane_points":
            # Do not silently destroy an existing metric calibration when only metadata is edited.
            profile["calibration"] = existing["calibration"]
            profile["image_size"] = existing.get("image_size")

    # Validate calibration at its own saved image resolution before writing.
    CameraCalibration(camera_view, profile, frame_size=(int((profile.get("image_size") or {}).get("width", 0)),
                                                         int((profile.get("image_size") or {}).get("height", 0)))
                      if profile.get("image_size") else None)

    path = _path(camera_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    return profile


def delete_camera_profile(camera_id: str) -> bool:
    path = _path(camera_id)
    if not path.is_file():
        return False
    path.unlink()
    return True
