"""Persistent job & saved-case storage (SQLite).

Replaces the old in-memory `jobs = {}` dict so analysis results survive a
backend restart, and gives saved cases a server-side home instead of only
living in one browser's localStorage (which doesn't survive a cleared
profile and isn't shared across devices).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "anchor_vi.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL;")
_conn.execute(
    """CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )"""
)
_conn.execute(
    """CREATE TABLE IF NOT EXISTS cases (
        id TEXT PRIMARY KEY,
        job_id TEXT,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )"""
)
_conn.commit()


def create_job(job_id: str, data: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, created_at, updated_at, payload) VALUES (?, ?, ?, ?)",
            (job_id, now, now, json.dumps(data)),
        )
        _conn.commit()


def update_job(job_id: str, **changes) -> None:
    with _lock:
        row = _conn.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return
        data = json.loads(row[0])
        data.update(changes)
        _conn.execute(
            "UPDATE jobs SET payload = ?, updated_at = ? WHERE job_id = ?",
            (json.dumps(data), datetime.now(timezone.utc).isoformat(), job_id),
        )
        _conn.commit()


def get_job(job_id: str) -> dict | None:
    with _lock:
        row = _conn.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return json.loads(row[0]) if row else None


def delete_job(job_id: str) -> bool:
    with _lock:
        cur = _conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        _conn.commit()
    return cur.rowcount > 0


def create_case(case: dict) -> dict:
    if not isinstance(case, dict):
        raise ValueError("Payload kasus harus berupa object JSON.")
    case_id = str(case.get("id") or "").strip()
    if not case_id:
        raise ValueError("Kasus harus memiliki field 'id'.")
    now = datetime.now(timezone.utc).isoformat()
    record = {**case, "savedAt": now}
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO cases (id, job_id, created_at, payload) VALUES (?, ?, ?, ?)",
            (case_id, case.get("jobId"), now, json.dumps(record)),
        )
        _conn.commit()
    return record


def create_case_from_job(job: dict) -> dict:
    """Persist a completed analysis as a dashboard case without a browser."""
    if not isinstance(job, dict) or job.get("status") != "completed":
        raise ValueError("Hanya job yang selesai yang dapat disimpan sebagai kasus.")
    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("Job selesai harus memiliki job_id.")

    metadata = job.get("metadata") or {}
    matrix = job.get("matrix_13") or {}
    primary_event = job.get("primary_event")
    country = metadata.get("country") or "Indonesia"
    flag_map = {"Indonesia": "🇮🇩", "Malaysia": "🇲🇾", "Thailand": "🇹🇭", "Vietnam": "🇻🇳"}
    crash_type = matrix.get("crash_type") or job.get("crash_type") or "Belum dapat diklasifikasikan"
    road_user_pair = matrix.get("road_user_pair") or "Mobil – Mobil"
    location = metadata.get("location_name") or "Lokasi CCTV tidak diketahui"

    case = {
        "id": f"KASUS-{job_id[:8].upper()}",
        "jobId": job_id,
        "title": f"{location} - {crash_type}",
        "location": location,
        "country": country,
        "flag": flag_map.get(country, "🇮🇩"),
        "date": datetime.now().astimezone().isoformat(),
        "roadUserPair": road_user_pair,
        "crashType": crash_type,
        "videoUrl": f"/api/results/{job_id}",
        "variables": {
            "crashType": crash_type,
            "roadUserPair": road_user_pair,
            "vehicle1Id": matrix.get("vehicle1_id"),
            "vehicle2Id": matrix.get("vehicle2_id"),
            "vehicle1Role": matrix.get("vehicle1_role"),
            "vehicle2Role": matrix.get("vehicle2_role"),
            "vehicle1Maneuver": matrix.get("vehicle1_maneuver"),
            "vehicle2Maneuver": matrix.get("vehicle2_maneuver"),
            "vehicle1Speed": matrix.get("vehicle1_display_speed_kmh", matrix.get("vehicle1_pre_crash_speed_kmh")),
            "vehicle2Speed": matrix.get("vehicle2_display_speed_kmh", matrix.get("vehicle2_pre_crash_speed_kmh")),
            "vehicle1AverageSpeed": matrix.get("vehicle1_average_speed_kmh"),
            "vehicle2AverageSpeed": matrix.get("vehicle2_average_speed_kmh"),
            "vehicle1SpeedSource": matrix.get("vehicle1_display_speed_source", matrix.get("vehicle1_speed_source", "unavailable")),
            "vehicle2SpeedSource": matrix.get("vehicle2_display_speed_source", matrix.get("vehicle2_speed_source", "unavailable")),
            "vehicle1SpeedSampleCount": matrix.get("vehicle1_display_speed_sample_count", matrix.get("vehicle1_speed_sample_count", 0)),
            "vehicle2SpeedSampleCount": matrix.get("vehicle2_display_speed_sample_count", matrix.get("vehicle2_speed_sample_count", 0)),
            "pairSpeedSource": matrix.get("pair_speed_source", "unavailable"),
            "pairSpeedSampleCount": matrix.get("pair_speed_sample_count", 0),
            "vehicle1SpeedDiagnostics": matrix.get("vehicle1_speed_diagnostics", {}),
            "vehicle2SpeedDiagnostics": matrix.get("vehicle2_speed_diagnostics", {}),
            "pairSpeedDiagnostic": matrix.get("pair_speed_diagnostic", ""),
            "preImpactWindow": matrix.get("pre_impact_window_seconds", 0.8),
            "vehicle1SpeedPx": matrix.get("vehicle1_pre_crash_speed_px_s"),
            "vehicle2SpeedPx": matrix.get("vehicle2_pre_crash_speed_px_s"),
            "relativeSpeed": matrix.get("relative_speed_kmh"),
            "relativeSpeedPx": matrix.get("relative_speed_px_s"),
            "impactSpeed": matrix.get("impact_speed_kmh"),
            "impactSpeedPx": matrix.get("impact_speed_px_s"),
            "impactAngle": matrix.get("impact_angle_deg"),
            "evasiveAction": matrix.get("evasive_action"),
            "pointOfImpact": matrix.get("point_of_impact"),
            "crashSeverity": matrix.get("crash_severity"),
            "severityCategory": matrix.get("severity_category"),
            "severityBasis": matrix.get("severity_basis"),
            "weather": matrix.get("weather_conditions") or metadata.get("weather"),
            "classificationConfidence": matrix.get("classification_confidence"),
            "roleAssignmentReason": matrix.get("role_assignment_reason"),
            "speedStatus": matrix.get("speed_status"),
            "speedWarning": matrix.get("speed_calibration_warning"),
        },
        "matrix13": matrix,
        "eventInfo": ({**primary_event, "classificationEvidence": primary_event.get("classification_evidence", [])}
                      if primary_event else None),
        "candidateEvents": job.get("candidate_events", []),
        "aseanNcapRelevance": job.get("suggested_protocols", []),
        "speedProfile": job.get("speed_profile", []),
        "vehicleSpeedSummaries": job.get("vehicle_speed_summaries", []),
        "vehicleSpeedProfiles": job.get("vehicle_speed_profiles", []),
        "calibration": job.get("calibration", {}),
        "pairDiagnostics": job.get("pair_diagnostics", []),
        "detectionDiagnostics": job.get("detection_diagnostics", {}),
        "trackletRecoveryDiagnostics": job.get("tracklet_recovery_diagnostics", []),
        "cameraAutoEstimate": job.get("camera_auto_estimate", {}),
        "autoCalibration": job.get("auto_calibration", {}),
        "analysisPipelineBuild": job.get("analysis_pipeline_build"),
        "backendApiVersion": job.get("api_version"),
        "analysisStatus": (
            "potential_contact_found" if job.get("candidate_event_count", 0) > 0
            else "no_candidate_contact"
        ),
    }
    return create_case(case)


def list_cases() -> list[dict]:
    # Backfill completed jobs created before automatic case persistence was
    # introduced. INSERT OR REPLACE keeps this safe and idempotent.
    with _lock:
        completed_rows = _conn.execute(
            "SELECT payload FROM jobs WHERE json_extract(payload, '$.status') = 'completed'"
        ).fetchall()
        existing_ids = {
            row[0] for row in _conn.execute("SELECT id FROM cases").fetchall()
        }
    for (payload,) in completed_rows:
        job = json.loads(payload)
        expected_id = f"KASUS-{str(job.get('job_id') or '')[:8].upper()}"
        if expected_id and expected_id not in existing_ids:
            create_case_from_job(job)

    with _lock:
        rows = _conn.execute("SELECT payload FROM cases ORDER BY created_at DESC").fetchall()
    return [json.loads(row[0]) for row in rows]


def get_case(case_id: str) -> dict | None:
    with _lock:
        row = _conn.execute("SELECT payload FROM cases WHERE id = ?", (case_id,)).fetchone()
    return json.loads(row[0]) if row else None


def delete_case(case_id: str) -> bool:
    with _lock:
        cur = _conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        _conn.commit()
    return cur.rowcount > 0
