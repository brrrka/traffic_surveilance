"""FastAPI backend for ANCHOR VI Matrix 13.

Run from the backend project directory:
    python -m uvicorn api:app --host 127.0.0.1 --port 8001 --reload
"""
import math
import json
import logging
import shutil
import subprocess
import traceback
import uuid

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detector import VehicleDetector
from trajectory import TrajectoryManager
from accident_detector import AccidentDetector
from visualizer import Visualizer
from matrix_builder import build_empty_matrix_13, build_matrix_13, build_speed_profile
from camera_calibration import CameraCalibration
from speed_engine import VehicleSpeedEngine, infer_camera_view, BUILD_ID as SPEED_ENGINE_BUILD_ID
from tracklet_recovery import recover_candidate, BUILD_ID as TRACKLET_RECOVERY_BUILD_ID
from camera_profiles import (
    delete_camera_profile, get_camera_profile, list_camera_profiles, save_camera_profile,
)
import job_store

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yolo11s.pt"
UPLOAD_DIR = ROOT / "data" / "input" / "api_uploads"
OUTPUT_DIR = ROOT / "data" / "output" / "api_results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 250 * 1024 * 1024

logger = logging.getLogger("anchor_vi")

app = FastAPI(title="ANCHOR VI Research API", version="3.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=1)


def update_job(job_id, **changes):
    job_store.update_job(job_id, **changes)


def _protocols_for_crash(crash_type_code):
    """Indicative protocol mapping by ANCHOR VI scenario code."""
    mapping = {
        "REAR_MOVING": ["FCW"],
        "TURN_ACROSS_PATH_ONCOMING": ["AEB-HeaDon"],
        "TURN_ACROSS_PATH_LATERAL": ["CTR"],
        "CROSSING": ["CTR"],
        "SAME_DIRECTION_TURN": ["FCW"],
        "CHANGE_LANE_ONCOMING": ["AEB-HeaDon", "FCW"],
        "CHANGE_LANE_OVERTAKING": ["FCW"],
    }
    return mapping.get(crash_type_code, [])


def run_analysis(job_id, input_path, output_path, metadata):
    cap = None
    writer = None
    processed = 0
    total_frames = 0
    try:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Model tidak ditemukan: {MODEL_PATH}")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Video tidak dapat dibuka oleh OpenCV.")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise RuntimeError(f"FPS video tidak valid: {fps}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0:
            raise RuntimeError("Resolusi video tidak valid.")

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("VideoWriter gagal membuat video output.")

        calibration = CameraCalibration(
            metadata.get("camera_view", "auto"),
            metadata.get("camera_profile") or metadata.get("calibration_profile"),
            frame_size=(width, height),
        )
        detector = VehicleDetector(str(MODEL_PATH), tracker=str(ROOT / "bytetrack.yaml"))
        trajectory = TrajectoryManager(fps, calibration=calibration)
        speed_engine = VehicleSpeedEngine(
            fps=fps, calibration=calibration,
            camera_view=metadata.get("camera_view", "auto"),
            mode=metadata.get("speed_mode", "both"),
        )
        accident_detector = AccidentDetector(fps=fps)
        visualizer = Visualizer()

        processed = 0
        candidate_events = []
        total_car_detections = 0
        frames_with_car = 0
        frames_with_two_or_more_cars = 0
        max_cars_per_frame = 0
        unique_track_ids = set()
        update_job(
            job_id, status="processing", progress=0,
            message="YOLO11s + ByteTrack + Speed Engine + tujuh skenario ANCHOR VI sedang berjalan",
            fps=round(fps, 3), width=width, height=height, total_frames=total_frames,
        )

        def append_event_record(event, histories_now):
            event["time_s"] = round(max(0, int(event["event_frame"]) - 1) / fps, 3)
            matrix = build_matrix_13(event, histories_now, fps, metadata, calibration)
            event_speed_profile = build_speed_profile(event, histories_now, fps)
            candidate_events.append({
                "event": event,
                "matrix_13": matrix,
                "speed_profile": event_speed_profile,
                "suggested_protocols": _protocols_for_crash(event.get("crash_type_code")),
            })

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            tracked = detector.detect_and_track(frame)
            histories = trajectory.update(tracked)
            speed_engine.update(histories, tracked, processed)

            car_ids_this_frame = set()
            for obj in tracked:
                if obj.get("class") != "car":
                    continue
                try:
                    tid = int(obj["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                car_ids_this_frame.add(tid)
                unique_track_ids.add(tid)
            total_car_detections += len(car_ids_this_frame)
            if car_ids_this_frame:
                frames_with_car += 1
            if len(car_ids_this_frame) >= 2:
                frames_with_two_or_more_cars += 1
            max_cars_per_frame = max(max_cars_per_frame, len(car_ids_this_frame))

            events = accident_detector.detect(histories, tracked)
            for event in events:
                append_event_record(event, histories)

            annotated = visualizer.draw_all(
                frame, tracked, histories, events,
                frame_number=processed, fps=fps,
            )
            writer.write(annotated)

            if processed % 10 == 0 or (total_frames and processed == total_frames):
                progress = min(98, round(processed * 100 / total_frames)) if total_frames else 0
                update_job(
                    job_id, progress=progress, processed_frames=processed,
                    message=f"Menganalisis frame {processed}/{total_frames or '?'}; "
                            f"kendaraan aktif={len(car_ids_this_frame)}",
                )

        if processed == 0:
            raise RuntimeError("Video kosong atau tidak dapat dibaca.")

        # Offline synchronized-frame pass: catches a brief overlap that may be
        # missed by the online persistence gate or a short track interruption.
        # It still requires geometric evidence and does not declare a proven crash.
        fallback_events = accident_detector.finalize(histories)
        known_pairs = {tuple(sorted(item["event"].get("pair_ids", []))) for item in candidate_events}
        for event in fallback_events:
            pair_key = tuple(sorted(event.get("pair_ids", [])))
            if pair_key and pair_key in known_pairs:
                continue
            append_event_record(event, histories)
            if pair_key:
                known_pairs.add(pair_key)

        # Post-contact: recover broken ByteTrack tracklets only for actual
        # geometric-contact candidates. Never claim identities are certain.
        tracklet_diagnostics = []
        for item in candidate_events:
            event = item["event"]
            histories_for_event = histories
            if not event.get("crash_type_code"):
                recovered, recovery_diag = recover_candidate(event, histories, accident_detector, fps)
                tracklet_diagnostics.append(recovery_diag)
                if recovered is not None:
                    improved_event, recovered_histories = recovered
                    item["event"] = improved_event
                    histories_for_event = recovered_histories
                    item["suggested_protocols"] = _protocols_for_crash(
                        improved_event.get("crash_type_code")
                    )
            # Recompute every candidate AFTER the video is processed, so the
            # average uses all observed frames, not only frames before event.
            # Pre-impact speed still uses only frames before the event.
            item["matrix_13"] = build_matrix_13(
                item["event"], histories_for_event, fps, metadata, calibration,
                vehicle_speed_samples=speed_engine.samples,
            )
            item["speed_profile"] = build_speed_profile(
                item["event"], histories_for_event, fps,
                vehicle_speed_samples=speed_engine.samples,
            )

        writer.release()
        writer = None
        cap.release()
        cap = None

        pair_diagnostics = accident_detector.get_diagnostics()
        detector_diagnostics_message = accident_detector.diagnostics_message()
        vehicle_speed_summaries = speed_engine.summaries()
        vehicle_speed_profiles = speed_engine.profiles()
        camera_auto_estimate = infer_camera_view(histories)
        any_metric_speed = any(row.get("speed_status") == "calibrated_reference" for row in vehicle_speed_summaries)
        any_prior_speed = any(row.get("speed_status") == "estimated_prior" for row in vehicle_speed_summaries)
        auto_calibration = {
            "mode": "auto",
            "camera_view_estimate": camera_auto_estimate,
            "metric_reference_available": bool(getattr(calibration, "H", None) is not None),
            "metric_speed_available": bool(any_metric_speed),
            "vehicle_dimension_prior_used": bool(any_prior_speed),
            "speed_output_policy": (
                "calibrated_kmh" if any_metric_speed else
                "low_confidence_vehicle_dimension_prior" if any_prior_speed else
                "pixel_speed_only"
            ),
            "warning": (
                "Kalibrasi bidang jalan dengan referensi meter tersedia." if any_metric_speed else
                "Sebagian kendaraan memakai prior dimensi generik berkeyakinan rendah; bukan kalibrasi metrik." if any_prior_speed else
                "Tidak ditemukan anchor meter otomatis yang dapat dipertanggungjawabkan; tampilkan px/s dan jangan mengarang km/jam."
            ),
        }
        detection_diagnostics = {
            "total_frames": int(total_frames or processed),
            "processed_frames": int(processed),
            "frames_with_at_least_one_car": int(frames_with_car),
            "frames_with_two_or_more_cars": int(frames_with_two_or_more_cars),
            "max_cars_in_one_frame": int(max_cars_per_frame),
            "total_car_detections_across_frames": int(total_car_detections),
            "unique_track_ids": sorted(int(x) for x in unique_track_ids),
            "unique_track_count": len(unique_track_ids),
            "pair_comparison_count": len(pair_diagnostics),
        }

        codec_note = "Output video MP4 (OpenCV codec mp4v)."
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            h264_path = output_path.with_name(output_path.stem + "_h264.mp4")
            command = [
                ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(output_path), "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(h264_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                if h264_path.is_file() and h264_path.stat().st_size > 0:
                    h264_path.replace(output_path)
                    codec_note = "Output video MP4 H.264 siap diputar di browser."
            except (subprocess.CalledProcessError, OSError):
                h264_path.unlink(missing_ok=True)
                codec_note = "Konversi H.264 gagal; file MP4 asli tetap disimpan."

        def event_score(item):
            ev = item["event"]
            evidence = ev.get("evidence", {})
            score = min(float(ev.get("bbox_iou", evidence.get("max_bbox_iou", 0)) or 0), 0.5) * 8
            score += min(int(evidence.get("pair_near_frames", 0)), 8) * 0.08
            if ev.get("crash_type_code"):
                score += 1.5
            if ev.get("classification_confidence") == "Sedang":
                score += 0.3
            return score

        candidate_events.sort(key=event_score, reverse=True)
        for item in candidate_events:
            item["candidate_score"] = round(event_score(item), 3)

        primary = candidate_events[0] if candidate_events else None
        # Only the two vehicles selected as the primary contact candidate
        # should appear in the dashboard's vehicle-speed summaries/profiles.
        # This does not discard other tracks from the internal computation.
        if primary:
            selected_ids = {int(primary["event"]["v1"]), int(primary["event"]["v2"])}
            vehicle_speed_summaries = [row for row in vehicle_speed_summaries
                                       if int(row.get("vehicle_id", -1)) in selected_ids]
            vehicle_speed_profiles = [row for row in vehicle_speed_profiles
                                      if int(row.get("vehicle_id", -1)) in selected_ids]
        else:
            vehicle_speed_summaries = []
            vehicle_speed_profiles = []
        if primary:
            matrix = primary["matrix_13"]
            crash_type = matrix["crash_type"]
            speed_profile = primary["speed_profile"]
            protocols = primary["suggested_protocols"]
            primary_event = primary["event"]
            analysis_message = f"{len(candidate_events)} kandidat kontak Car-Car ditemukan."
        else:
            matrix = build_empty_matrix_13(metadata, calibration)
            crash_type = matrix["crash_type"]
            speed_profile = []
            protocols = []
            primary_event = None
            analysis_message = (
                "Video selesai diproses, tetapi tidak ada pasangan yang memenuhi bukti kandidat kontak. "
                + detector_diagnostics_message
                + " Ringkasan kecepatan kendaraan tetap tersedia secara terpisah."
            )

        update_job(
            job_id,
            status="completed", progress=100, processed_frames=processed,
            message=analysis_message, video_url=f"/api/results/{job_id}",
            video_codec_note=codec_note, candidate_event_count=len(candidate_events),
            candidate_events=candidate_events, primary_event=primary_event,
            matrix_13=matrix, speed_profile=speed_profile,
            vehicle_speed_summaries=vehicle_speed_summaries,
            vehicle_speed_profiles=vehicle_speed_profiles,
            suggested_protocols=protocols, crash_type=crash_type,
            metadata=metadata, calibration=calibration.describe(),
            camera_auto_estimate=camera_auto_estimate,
            auto_calibration=auto_calibration,
            camera_id=metadata.get("camera_id") or None,
            camera_name=metadata.get("camera_name") or None,
            pair_diagnostics=pair_diagnostics,
            detector_diagnostics_message=detector_diagnostics_message,
            detection_diagnostics=detection_diagnostics,
            tracklet_recovery_diagnostics=tracklet_diagnostics,
            selected_primary_reason=(
                "Kandidat dengan bukti geometri terkuat; kandidat kontak bukan bukti final tabrakan fisik."
            ),
            speed_engine_build=SPEED_ENGINE_BUILD_ID,
            analysis_pipeline_build="ANCHOR-VI-pipeline-3.4",
            tracklet_recovery_build=TRACKLET_RECOVERY_BUILD_ID,
        )

    except Exception as exc:
        # Keep the complete traceback in the backend terminal and in the job
        # record. This is important because str(KeyError(24)) only displays "24".
        error_traceback = traceback.format_exc()
        error_type = type(exc).__name__
        error_detail = repr(exc)
        error_message = f"{error_type}: {exc}" if str(exc) else error_type

        logger.error(
            "[ANCHOR VI] Analysis failed | job_id=%s | processed=%s/%s | %s",
            job_id,
            processed,
            total_frames,
            error_detail,
        )
        logger.error("%s", error_traceback)

        update_job(
            job_id,
            status="failed",
            # Do not reset progress: retain the last known progress value.
            processed_frames=int(processed),
            message=error_message,
            error_type=error_type,
            error_detail=error_detail,
            error_traceback=error_traceback,
        )

        # Cleanup must never hide the original analysis exception.
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "[ANCHOR VI] Could not remove partial output | job_id=%s | path=%s",
                job_id,
                output_path,
            )
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.exception("[ANCHOR VI] Failed to release VideoCapture | job_id=%s", job_id)
        if writer is not None:
            try:
                writer.release()
            except Exception:
                logger.exception("[ANCHOR VI] Failed to release VideoWriter | job_id=%s", job_id)
        # The raw upload is never needed again after this point (the dashboard
        # only plays back output_path); remove it so public uploads don't
        # accumulate disk usage indefinitely.
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("[ANCHOR VI] Failed to remove uploaded input | job_id=%s | path=%s", job_id, input_path)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
        "camera_profile_count": len(list_camera_profiles()),
        "api_version": app.version,
        "contact_detector_build": AccidentDetector.BUILD_ID,
        "speed_engine_build": SPEED_ENGINE_BUILD_ID,
        "analysis_pipeline_build": "ANCHOR-VI-pipeline-3.4",
        "tracklet_recovery_build": TRACKLET_RECOVERY_BUILD_ID,
        "camera_calibration_mode": "auto_with_explicit_metric_uncertainty",
        "crash_scenario_count": 7,
    }


@app.get("/api/cameras")
def cameras_list():
    """List CCTV profiles stored locally on this backend."""
    return {"cameras": list_camera_profiles()}


@app.get("/api/cameras/{camera_id}")
def camera_detail(camera_id: str):
    profile = get_camera_profile(camera_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profil CCTV tidak ditemukan.")
    return profile


@app.put("/api/cameras/{camera_id}")
def camera_upsert(camera_id: str, payload: dict = Body(...)):
    """Create/update one camera profile and optional metric ground-plane calibration."""
    try:
        profile = save_camera_profile(camera_id, payload)
        # Return the calibration summary after validation.
        saved_size = profile.get("image_size") or {}
        saved_frame_size = (int(saved_size["width"]), int(saved_size["height"])) if saved_size.get("width") and saved_size.get("height") else None
        cal = CameraCalibration(profile["camera_view"], profile, frame_size=saved_frame_size)
        return {"profile": profile, "calibration": cal.describe()}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.delete("/api/cameras/{camera_id}")
def camera_delete(camera_id: str):
    try:
        deleted = delete_camera_profile(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not deleted:
        raise HTTPException(status_code=404, detail="Profil CCTV tidak ditemukan.")
    return {"deleted": True, "camera_id": camera_id}


@app.post("/api/analyze", status_code=202)
def analyze(
    video: UploadFile = File(...),
    location_name: str = Form("Unknown CCTV location"),
    country: str = Form("Indonesia"),
    weather: str = Form("Unknown"),
    lighting: str = Form("Unknown"),
    road_surface: str = Form("Unknown"),
    severity_reference: str = Form("Unknown"),
    camera_view: str = Form("auto"),
    camera_id: str = Form(""),
    calibration_profile: str = Form(""),
    speed_mode: str = Form("both"),
    calibration_mode: str = Form("auto"),
):
    # ANCHOR VI defaults to AUTO camera view and BOTH speed outputs. A saved
    # per-camera metric profile is still honored when explicitly selected;
    # otherwise no manual drawing is requested from the dashboard user.
    if speed_mode not in {"both", "all", "event"}:
        raise HTTPException(status_code=422, detail="speed_mode harus both, all, atau event.")
    if calibration_mode != "auto":
        raise HTTPException(status_code=422, detail="Versi ini menggunakan calibration_mode=auto.")
    camera_profile = None
    if camera_id.strip():
        try:
            camera_profile = get_camera_profile(camera_id.strip())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if camera_profile is None:
            raise HTTPException(status_code=404, detail=f"Profil CCTV '{camera_id}' tidak ditemukan.")
        camera_view = camera_profile.get("camera_view", camera_view)
    try:
        inline_profile = json.loads(calibration_profile) if calibration_profile.strip() else None
        calibration_preview = CameraCalibration(camera_view, camera_profile or inline_profile)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f'Profil kalibrasi tidak valid: {exc}')
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Format video harus mp4, avi, mov, atau mkv.")

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    size = 0

    try:
        with input_path.open("wb") as dest:
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Ukuran video maksimal 250 MB.")
                dest.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        video.file.close()

    if size == 0:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File video kosong.")

    metadata = {
        "location_name": location_name,
        "country": country,
        "weather": weather,
        "lighting": lighting,
        "road_surface": road_surface,
        "severity_reference": severity_reference,
        "camera_view": camera_view,
        "speed_mode": speed_mode,
        "calibration_mode": calibration_mode,
        "camera_id": camera_id.strip() or None,
        "camera_name": camera_profile.get("camera_name") if camera_profile else None,
        "camera_profile": camera_profile,
        "calibration_profile": inline_profile,
    }

    job_store.create_job(job_id, {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Menunggu giliran pemrosesan",
        "error_type": None,
        "error_detail": None,
        "error_traceback": None,
        "fps": None,
        "processed_frames": 0,
        "video_url": None,
        "candidate_event_count": 0,
        "candidate_events": [],
        "matrix_13": build_empty_matrix_13(metadata),
        "speed_profile": [],
        "suggested_protocols": [],
        "metadata": metadata,
        "calibration": calibration_preview.describe(),
        "camera_id": camera_id.strip() or None,
        "camera_name": camera_profile.get("camera_name") if camera_profile else None,
        "vehicle_speed_summaries": [],
        "vehicle_speed_profiles": [],
        "pair_diagnostics": [],
        "tracklet_recovery_diagnostics": [],
        "detection_diagnostics": {},
        "camera_auto_estimate": {"label": "Menunggu analisis", "confidence": "Tidak tersedia", "is_metric_calibration": False},
        "auto_calibration": {"mode": "auto", "speed_output_policy": "pending"},
        "detector_diagnostics_message": "Diagnostik pasangan akan tersedia setelah pemrosesan selesai.",
        "speed_engine_build": SPEED_ENGINE_BUILD_ID,
        "analysis_pipeline_build": "ANCHOR-VI-pipeline-3.4",
        "tracklet_recovery_build": TRACKLET_RECOVERY_BUILD_ID,
    })

    executor.submit(run_analysis, job_id, input_path, output_path, metadata)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job tidak ditemukan.")
    return job


@app.get("/api/results/{job_id}")
def video_result(job_id: str):
    job = job_store.get_job(job_id)
    if job is None or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Hasil video belum tersedia.")
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File video hasil tidak ditemukan.")
    return FileResponse(str(path), media_type="video/mp4", filename=f"ANCHOR_VI_{job_id}.mp4")



@app.get("/api/jobs/{job_id}/diagnostics")
def job_diagnostics(job_id: str):
    """Return pair-level contact diagnostics for debugging candidate detection."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job tidak ditemukan.")
    return {
            "job_id": job_id,
            "status": job.get("status"),
            "candidate_event_count": job.get("candidate_event_count", 0),
            "summary": job.get("detector_diagnostics_message"),
            "pairs": job.get("pair_diagnostics", []),
            "tracklet_recovery": job.get("tracklet_recovery_diagnostics", []),
            "detection": job.get("detection_diagnostics", {}),
            "camera_auto_estimate": job.get("camera_auto_estimate", {}),
            "auto_calibration": job.get("auto_calibration", {}),
            "vehicle_speed_summaries": job.get("vehicle_speed_summaries", []),
            "primary_speed_diagnostics": {
                "v1": (job.get("matrix_13") or {}).get("vehicle1_speed_diagnostics", {}),
                "v2": (job.get("matrix_13") or {}).get("vehicle2_speed_diagnostics", {}),
                "pair": (job.get("matrix_13") or {}).get("pair_speed_vector_diagnostics", {}),
                "pair_message": (job.get("matrix_13") or {}).get("pair_speed_diagnostic", ""),
                "position_alignment": (job.get("matrix_13") or {}).get("pair_speed_position_alignment"),
                "sync_method": (job.get("matrix_13") or {}).get("pair_speed_sync_method"),
            },
            "candidate_speed_diagnostics": [
                {
                    "v1": item.get("event", {}).get("v1"),
                    "v2": item.get("event", {}).get("v2"),
                    "event_frame": item.get("event", {}).get("event_frame"),
                    "crash_type": item.get("event", {}).get("crash_type_code"),
                    "score": item.get("candidate_score"),
                    "v1_speed": item.get("matrix_13", {}).get("vehicle1_display_speed_kmh"),
                    "v2_speed": item.get("matrix_13", {}).get("vehicle2_display_speed_kmh"),
                    "relative_speed": item.get("matrix_13", {}).get("relative_speed_kmh"),
                    "impact_speed": item.get("matrix_13", {}).get("impact_speed_kmh"),
                    "pair_status": item.get("matrix_13", {}).get("pair_speed_status"),
                    "pair_message": item.get("matrix_13", {}).get("pair_speed_diagnostic"),
                }
                for item in job.get("candidate_events", [])
            ],
            "analysis_pipeline_build": job.get("analysis_pipeline_build"),
        }

@app.get("/api/jobs/{job_id}/export")
def export_result(job_id: str):
    """Download analysis JSON; excludes video path and uploaded file bytes."""
    from fastapi.responses import JSONResponse
    job = job_store.get_job(job_id)
    if job is None or job.get('status') != 'completed':
        raise HTTPException(status_code=404, detail='Hasil analisis belum tersedia.')
    return JSONResponse(job)


@app.get("/api/cases")
def cases_list():
    """List saved crash-database cases, newest first (server-side, survives browser/device changes)."""
    return {"cases": job_store.list_cases()}


@app.post("/api/cases", status_code=201)
def cases_create(case: dict = Body(...)):
    try:
        return job_store.create_case(case)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/cases/{case_id}")
def cases_detail(case_id: str):
    record = job_store.get_case(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Kasus tidak ditemukan.")
    return record


@app.delete("/api/cases/{case_id}")
def cases_delete(case_id: str):
    if not job_store.delete_case(case_id):
        raise HTTPException(status_code=404, detail="Kasus tidak ditemukan.")
    return {"deleted": True, "case_id": case_id}


# Serve the Dashboard production build (Dashboard/dist) from this same
# process/port so FE and BE share one origin in deployment. Must be mounted
# last: Starlette matches routes in registration order, so every /api/*
# route above still takes priority over this catch-all.
DASHBOARD_DIST = ROOT.parent / "Dashboard" / "dist"
if DASHBOARD_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIST), html=True), name="dashboard")
