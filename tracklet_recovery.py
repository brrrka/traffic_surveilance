"""ANCHOR VI: conservative offline recovery of fragmented ByteTrack tracklets.

Runs AFTER YOLO11s + ByteTrack, never substitutes those models. Geometric
association is only a hypothesis, not verified identity. Never stitch tracklets
with overlapping frame ranges. Mark uncertain and retain audit trails.
"""
from __future__ import annotations

import math
from statistics import median

from crash_classifier import RuleBasedCrashClassifier

BUILD_ID = "ANCHOR-VI-tracklet-recovery-1.1"
ALIGNED = ("frame_indices", "trajectory", "footpoint_trajectory", "bbox_history",
           "analysis_trajectory", "world_trajectory")


def _normalise_histories(histories):
    """Return histories keyed by integer ByteTrack IDs.

    Invalid keys are skipped and duplicate IDs after integer conversion are
    reported rather than silently overwriting one tracklet with another.
    """
    normalized = {}
    invalid_keys = []
    duplicate_ids = set()

    for raw_id, history in histories.items():
        try:
            track_id = int(raw_id)
        except (TypeError, ValueError, OverflowError):
            invalid_keys.append(str(raw_id))
            continue

        if track_id in normalized:
            duplicate_ids.add(track_id)
            continue

        normalized[track_id] = history

    return normalized, invalid_keys, sorted(duplicate_ids)


def _history_window(history, event_frame):
    """Compact temporal diagnostics without using post-contact observations."""
    frames = history.get("frame_indices") or []
    valid_frames = []
    for frame in frames:
        try:
            valid_frames.append(int(frame))
        except (TypeError, ValueError, OverflowError):
            continue

    pre_frames = [frame for frame in valid_frames if frame <= event_frame]
    return {
        "first_frame": min(valid_frames) if valid_frames else None,
        "last_frame_before_event": max(pre_frames) if pre_frames else None,
        "observations_before_event": len(pre_frames),
    }


def _rows(hist, key):
    frames = hist.get("frame_indices") or []
    values = hist.get(key) or []
    return list(zip(frames, values)) if len(frames) == len(values) else []


def _diag(box):
    return math.hypot(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def _vel(points, frames, tail=True):
    if len(frames) < 3 or len(points) < 3:
        return None
    pts = list(points[-7:] if tail else points[:7])
    ff = list(frames[-7:] if tail else frames[:7])
    a, b = ff[0], ff[-1]
    if b <= a:
        return None
    return ((float(pts[-1][0]) - float(pts[0][0])) / (b - a),
            (float(pts[-1][1]) - float(pts[0][1])) / (b - a))


def _link_score(old, new, max_gap_frames):
    frames_a = old.get("frame_indices") or []
    frames_b = new.get("frame_indices") or []
    pa, pb = old.get("trajectory") or [], new.get("trajectory") or []
    ba, bb = old.get("bbox_history") or [], new.get("bbox_history") or []
    if not (frames_a and frames_b and len(pa) == len(frames_a)
            and len(pb) == len(frames_b) and len(ba) == len(frames_a)
            and len(bb) == len(frames_b)):
        return None
    gap = int(frames_b[0]) - int(frames_a[-1])
    if not (1 <= gap <= max_gap_frames):
        return None  # never merge observations in overlapping frames
    dia = max(1, _diag(ba[-1])); dib = max(1, _diag(bb[0]))
    scale = dib / dia
    if not 0.5 <= scale <= 2.0:
        return None
    va = _vel(pa, frames_a)
    vb = _vel(pb, frames_b, tail=False)
    if va is not None and vb is not None:
        norm_a, norm_b = math.hypot(*va), math.hypot(*vb)
        if norm_a > 1.2 and norm_b > 1.2:
            cosine = (va[0]*vb[0]+va[1]*vb[1])/(norm_a*norm_b)
            if cosine < 0.25:
                return None
    # Extrapolation should be conservative: do not extrapolate long distances
    # from a poorly observed fragment, whose velocity is likely unstable.
    use_prediction = va is not None and gap <= 8
    dx = va[0] * gap if use_prediction else 0.0
    dy = va[1] * gap if use_prediction else 0.0
    dist = math.hypot(pa[-1][0]+dx-pb[0][0], pa[-1][1]+dy-pb[0][1])
    ratio = dist / max(dia, dib)
    limit = 0.55 if use_prediction else 0.42
    if ratio > limit:
        return None
    # Penalize poor size consistency and long gaps.
    score = ratio + 0.14 * abs(math.log(scale)) + 0.008 * gap
    return {"score": round(score, 5), "gap_frames": gap,
            "endpoint_error_px": round(dist, 2), "normalized_error": round(ratio, 3),
            "bbox_scale_ratio": round(scale, 3)}


def recover_one(histories, target_id, excluded_ids=(), max_gap_frames=15, max_hops=5):
    """Reconstruct one endpoint tracklet, using predecessor tracklets only.

    Excludes the other collision candidate's ID so the two cars cannot be
    merged together. A near-tie is rejected as ambiguous. Missing target IDs
    are returned as diagnostics instead of raising KeyError.
    """
    normalized, invalid_keys, duplicate_ids = _normalise_histories(histories)
    target_id = int(target_id)

    base_info = {
        "target_id": target_id,
        "track_ids_chronological": [target_id],
        "links": [],
        "merged_observations": 0,
        "recovered": False,
    }

    if duplicate_ids:
        return None, {
            **base_info,
            "status": "ambiguous_normalized_track_ids",
            "duplicate_ids": duplicate_ids,
        }

    if target_id not in normalized:
        return None, {
            **base_info,
            "status": "target_tracklet_not_available",
            "invalid_history_keys": invalid_keys,
        }

    ids = [target_id]
    used = set(map(int, excluded_ids)) | {target_id}
    links = []

    while len(ids) <= max_hops:
        current_id = ids[0]
        current = normalized.get(current_id)
        if current is None:
            # Defensive guard: never let an incomplete recovery chain crash
            # the main video-analysis pipeline.
            return None, {
                **base_info,
                "track_ids_chronological": list(ids),
                "links": links,
                "status": "predecessor_tracklet_missing",
            }

        scored = []
        for tid, previous in normalized.items():
            if tid in used or previous.get("class") != "car":
                continue
            measure = _link_score(previous, current, max_gap_frames)
            if measure is not None:
                scored.append((measure["score"], tid, measure))

        scored.sort(key=lambda row: (row[0], row[1]))
        if not scored:
            break

        best = scored[0]
        if len(scored) > 1 and scored[1][0] - best[0] < 0.10:
            break  # unreliable identity, don't guess

        tid = best[1]
        used.add(tid)
        ids.insert(0, tid)
        links.insert(0, {"from_id": tid, "to_id": ids[1], **best[2]})

    parts = [normalized[tid] for tid in ids]
    merged = dict(normalized[target_id])
    if len(ids) > 1:
        for key in ALIGNED:
            merged[key] = [x for part in parts for x in (part.get(key) or [])]
        merged["speed_history"] = [
            x for part in parts for x in (part.get("speed_history") or [])
        ]
        merged["observations"] = len(merged.get("frame_indices", []))
        merged["track_id"] = target_id

    return merged, {
        "target_id": target_id,
        "track_ids_chronological": ids,
        "links": links,
        "merged_observations": len(merged.get("frame_indices") or []),
        "recovered": len(ids) > 1,
        "status": "recovered" if len(ids) > 1 else "no_predecessor_found",
    }


def recover_candidate(event, histories, detector, fps):
    """Try a low-confidence reclassification for an existing contact candidate.

    Does not synthesize contact, invent metric velocity, or claim tracklet
    identity is verified. Reconstructed trajectory data remains inspectable.
    Post-contact samples are excluded before any association is attempted.
    """
    pair = [int(event["v1"]), int(event["v2"])]
    event_frame = int(event["event_frame"])
    warning = (
        "Tracklet linking adalah hipotesis geometri, bukan identitas yang "
        "terverifikasi."
    )

    normalized, invalid_keys, duplicate_ids = _normalise_histories(histories)
    if pair[0] == pair[1]:
        return None, {
            "pair_ids": pair,
            "event_frame": event_frame,
            "status": "invalid_pair",
            "reason": "vehicle_ids_are_identical",
        }

    missing_from_histories = [tid for tid in pair if tid not in normalized]
    if missing_from_histories:
        return None, {
            "pair_ids": pair,
            "event_frame": event_frame,
            "status": "invalid_pair",
            "missing_ids": missing_from_histories,
            "invalid_history_keys": invalid_keys,
        }

    if duplicate_ids:
        return None, {
            "pair_ids": pair,
            "event_frame": event_frame,
            "status": "ambiguous_history_ids",
            "duplicate_ids": duplicate_ids,
        }

    # Build a strictly pre-contact view. Use ANY frame <= event_frame instead
    # of assuming frame_indices[0] is always the earliest valid observation.
    available = {}
    truncation_errors = []
    no_precontact_ids = []

    for track_id, history in normalized.items():
        window = _history_window(history, event_frame)
        if window["observations_before_event"] == 0:
            no_precontact_ids.append(track_id)
            continue

        try:
            truncated = detector._truncate_history(history, event_frame)
        except Exception as exc:
            truncation_errors.append({
                "track_id": track_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            continue

        truncated_frames = (truncated or {}).get("frame_indices") or []
        if not truncated_frames:
            no_precontact_ids.append(track_id)
            continue

        available[track_id] = truncated

    # This is the exact guard missing in version 1.0. A pair can exist in the
    # full histories but disappear from the pre-contact dictionary.
    missing_before_event = [tid for tid in pair if tid not in available]
    if missing_before_event:
        vehicle_info = {}
        for label, track_id in (("vehicle1", pair[0]), ("vehicle2", pair[1])):
            history = normalized.get(track_id, {})
            window = _history_window(history, event_frame)
            vehicle_info[label] = {
                "target_id": track_id,
                "recovered": False,
                "status": (
                    "no_precontact_history"
                    if window["observations_before_event"] == 0
                    else "history_truncation_failed_or_empty"
                ),
                **window,
            }

        return None, {
            "pair_ids": pair,
            "event_frame": event_frame,
            **vehicle_info,
            "status": "candidate_tracklet_missing_before_event_frame",
            "missing_pair_ids": missing_before_event,
            "excluded_tracklet_count": len(no_precontact_ids),
            "truncation_errors": truncation_errors,
            "warning": warning,
        }

    merged1, info1 = recover_one(
        available, pair[0], excluded_ids=(pair[1],)
    )
    if merged1 is None:
        return None, {
            "pair_ids": pair,
            "event_frame": event_frame,
            "vehicle1": info1,
            "status": "vehicle1_recovery_unavailable",
            "truncation_errors": truncation_errors,
            "warning": warning,
        }

    excluded_for_v2 = tuple(info1["track_ids_chronological"])
    merged2, info2 = recover_one(
        available, pair[1], excluded_ids=excluded_for_v2
    )
    diagnostic = {
        "pair_ids": pair,
        "event_frame": event_frame,
        "vehicle1": info1,
        "vehicle2": info2,
        "status": "not_recovered",
        "excluded_tracklet_count": len(no_precontact_ids),
        "truncation_errors": truncation_errors,
        "warning": warning,
    }
    if merged2 is None:
        diagnostic["status"] = "vehicle2_recovery_unavailable"
        return None, diagnostic

    if not info1["recovered"] and not info2["recovered"]:
        return None, diagnostic
    if min(merged1.get("observations", 0), merged2.get("observations", 0)) < 7:
        diagnostic["status"] = "insufficient_trajectory_after_linking"
        return None, diagnostic
    c = RuleBasedCrashClassifier().classify(merged1, merged2, pair[0], pair[1])
    diagnostic["classification_rule"] = c.get("rule_reason")
    diagnostic["suggested_code"] = c.get("crash_type_code")
    if not c.get("crash_type_code"):
        diagnostic["status"] = "still_unclassified"
        return None, diagnostic
    diagnostic["status"] = "classified_with_uncertain_tracklet_linking"
    result = dict(event)
    result.update({
        "v1": c["vehicle1_id"], "v2": c["vehicle2_id"],
        "crash_type": c["crash_type"], "crash_type_code": c["crash_type_code"],
        "classification_confidence": "Rendah",
        "maneuver_v1": c.get("maneuver_v1"), "maneuver_v2": c.get("maneuver_v2"),
        "vehicle1_role": c.get("vehicle1_role"), "vehicle2_role": c.get("vehicle2_role"),
        "role_assignment_reason": c.get("role_assignment_reason"),
        "impact_angle_deg": c.get("approach_angle_deg"),
        "impact_angle_reference": c.get("impact_angle_reference", "image_plane"),
        "classification_coordinate_space": c.get("classification_coordinate_space", "image_px"),
        "contact_side_v1": c.get("contact_side_v1"), "contact_side_v2": c.get("contact_side_v2"),
        "rule_reason": c.get("rule_reason"),
        "heading_v1_deg": c.get("heading_v1_deg"), "heading_v2_deg": c.get("heading_v2_deg"),
        "approach_heading_v1_deg": c.get("approach_heading_v1_deg"),
        "approach_heading_v2_deg": c.get("approach_heading_v2_deg"),
        "classification_evidence": list(c.get("classification_evidence") or []) + [
            "Lintasan direkonstruksi dari beberapa ID ByteTrack secara geometris; perlu verifikasi visual."
        ],
        "tracklet_recovery": diagnostic,
    })
    reconstructed = dict(normalized)
    reconstructed[pair[0]] = merged1
    reconstructed[pair[1]] = merged2
    return (result, reconstructed), diagnostic
