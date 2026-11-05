"""
ANCHOR VI - robust Car-Car potential-contact detector (v3.2).

This module separates:
1) potential-contact detection (bounding boxes + short-term trajectory geometry),
2) seven-scenario classification (RuleBasedCrashClassifier).

Important:
- A candidate is NOT proof of a collision.
- Pixel-space geometry can identify visual proximity, but cannot produce metric speed.
- Diagnostics are retained even when no candidate is emitted, so the backend
  can explain which gate rejected a vehicle pair.
"""
from __future__ import annotations

import math
from typing import Any

from crash_classifier import RuleBasedCrashClassifier


class AccidentDetector:
    BUILD_ID = "ANCHOR-VI-contact-detector-3.3"

    def __init__(
        self,
        fps: float = 30.0,
        min_observations: int = 7,
        min_iou: float = 0.001,
        strong_iou: float = 0.001,
        proximity_ratio: float = 0.30,
        strong_gap_ratio: float = 0.18,
        min_near_frames: int = 2,
        reset_separation_frames: int = 12,
        min_closing_px_s: float = 0.5,
        prediction_horizon_s: float = 0.75,
        predicted_cpa_ratio: float = 0.18,
    ):
        self.fps = float(fps)
        if not math.isfinite(self.fps) or not (1.0 <= self.fps <= 240.0):
            self.fps = 30.0

        self.min_observations = max(3, int(min_observations))
        self.min_iou = max(0.0, float(min_iou))
        self.strong_iou = max(self.min_iou, float(strong_iou))
        self.proximity_ratio = max(0.05, float(proximity_ratio))
        self.strong_gap_ratio = max(0.03, min(float(strong_gap_ratio), self.proximity_ratio))
        self.min_near_frames = max(1, int(min_near_frames))
        self.reset_separation_frames = max(2, int(reset_separation_frames))
        self.min_closing_px_s = max(0.0, float(min_closing_px_s))
        self.prediction_horizon_s = max(0.1, float(prediction_horizon_s))
        self.predicted_cpa_ratio = max(0.03, float(predicted_cpa_ratio))

        self.frame_index = 0
        self.pair_states: dict[tuple[int, int], dict[str, Any]] = {}
        self.pair_diagnostics: dict[tuple[int, int], dict[str, Any]] = {}
        self.classifier = RuleBasedCrashClassifier()

    @staticmethod
    def geometry(bbox1, bbox2):
        x1, y1, x2, y2 = map(float, bbox1)
        a1, b1, a2, b2 = map(float, bbox2)

        left, right = max(x1, a1), min(x2, a2)
        top, bottom = max(y1, b1), min(y2, b2)
        iw, ih = max(0.0, right - left), max(0.0, bottom - top)
        intersection = iw * ih

        area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area2 = max(0.0, a2 - a1) * max(0.0, b2 - b1)
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0.0

        gap_x = max(0.0, a1 - x2, x1 - a2)
        gap_y = max(0.0, b1 - y2, y1 - b2)
        gap = math.hypot(gap_x, gap_y)

        diag1 = math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))
        diag2 = math.hypot(max(0.0, a2 - a1), max(0.0, b2 - b1))
        small_diag = max(1.0, min(diag1, diag2))

        if intersection > 0:
            location = (
                int((left + right) / 2.0),
                int((top + bottom) / 2.0),
            )
        else:
            px1 = min(max((a1 + a2) / 2.0, x1), x2)
            py1 = min(max((b1 + b2) / 2.0, y1), y2)
            px2 = min(max((x1 + x2) / 2.0, a1), a2)
            py2 = min(max((y1 + y2) / 2.0, b1), b2)
            location = (int((px1 + px2) / 2.0), int((py1 + py2) / 2.0))

        return float(iou), float(gap), float(small_diag), location

    def _estimate_image_velocity(self, history):
        """
        Estimate centre-point image velocity from recent centre observations.
        This avoids mixing bbox-centre positions with footpoint velocities.
        """
        points = history.get("trajectory") or []
        frames = history.get("frame_indices") or []
        n = min(len(points), len(frames), 9)
        if n >= 3:
            pts = points[-n:]
            frs = frames[-n:]
            t = [(float(f) - float(frs[0])) / self.fps for f in frs]
            if t[-1] - t[0] > 1e-6:
                mt = sum(t) / len(t)
                denom = sum((v - mt) ** 2 for v in t)
                if denom > 1e-9:
                    mx = sum(float(p[0]) for p in pts) / len(pts)
                    my = sum(float(p[1]) for p in pts) / len(pts)
                    vx = sum((tt - mt) * (float(p[0]) - mx)
                             for tt, p in zip(t, pts)) / denom
                    vy = sum((tt - mt) * (float(p[1]) - my)
                             for tt, p in zip(t, pts)) / denom
                    return float(vx), float(vy), "trajectory_regression"

        # Backwards-compatible fallback; these values are px/s in current
        # TrajectoryManager versions.
        return (
            float(history.get("velocity_x") or 0.0),
            float(history.get("velocity_y") or 0.0),
            "trajectory_manager_velocity",
        )

    @staticmethod
    def _predict_closest_approach(center1, center2, velocity1, velocity2, horizon_s):
        rx = float(center2[0]) - float(center1[0])
        ry = float(center2[1]) - float(center1[1])
        rvx = float(velocity2[0]) - float(velocity1[0])
        rvy = float(velocity2[1]) - float(velocity1[1])
        vv = rvx * rvx + rvy * rvy
        distance = math.hypot(rx, ry)
        if vv <= 1e-9:
            return None, distance, 0.0
        t_cpa = -(rx * rvx + ry * rvy) / vv
        if t_cpa < 0.0 or t_cpa > horizon_s:
            return t_cpa, distance, 0.0
        cpa = math.hypot(rx + rvx * t_cpa, ry + rvy * t_cpa)
        closing = max(0.0, -(rx * rvx + ry * rvy) / max(distance, 1e-9))
        return float(t_cpa), float(cpa), float(closing)

    def _diagnostic_record(self, pair):
        return self.pair_diagnostics.setdefault(pair, {
            "vehicle_ids": [int(pair[0]), int(pair[1])],
            "frames_compared": 0,
            "min_observations_v1": None,
            "min_observations_v2": None,
            "min_bbox_gap_px": None,
            "min_normalized_gap": None,
            "max_bbox_iou": 0.0,
            "max_closing_speed_px_s": 0.0,
            "best_predicted_cpa_px": None,
            "best_predicted_cpa_time_s": None,
            "max_near_frames": 0,
            "candidate_emitted": False,
            "last_reason": "Pasangan belum dievaluasi.",
            "last_frame": None,
        })

    def _update_diagnostic(self, pair, obs1, obs2, iou, gap, diag,
                           closing, cpa_px, cpa_time, near_frames,
                           emitted, reason):
        d = self._diagnostic_record(pair)
        d["frames_compared"] += 1
        d["min_observations_v1"] = obs1 if d["min_observations_v1"] is None else min(d["min_observations_v1"], obs1)
        d["min_observations_v2"] = obs2 if d["min_observations_v2"] is None else min(d["min_observations_v2"], obs2)
        d["min_bbox_gap_px"] = round(gap, 2) if d["min_bbox_gap_px"] is None else min(d["min_bbox_gap_px"], round(gap, 2))
        ratio = gap / max(diag, 1.0)
        d["min_normalized_gap"] = round(ratio, 4) if d["min_normalized_gap"] is None else min(d["min_normalized_gap"], round(ratio, 4))
        d["max_bbox_iou"] = max(float(d["max_bbox_iou"]), round(iou, 5))
        d["max_closing_speed_px_s"] = max(float(d["max_closing_speed_px_s"]), round(closing, 2))
        d["max_near_frames"] = max(int(d["max_near_frames"]), int(near_frames))
        if cpa_px is not None and (d["best_predicted_cpa_px"] is None or cpa_px < d["best_predicted_cpa_px"]):
            d["best_predicted_cpa_px"] = round(cpa_px, 2)
            d["best_predicted_cpa_time_s"] = round(cpa_time, 3) if cpa_time is not None else None
        d["candidate_emitted"] = bool(d["candidate_emitted"] or emitted)
        d["last_reason"] = reason
        d["last_frame"] = self.frame_index

    def _speed_readiness_at_event(self, history):
        """Audit-only snapshot. Does not change event timing or classification."""
        event_frame = int(self.frame_index)
        start_08 = event_frame - max(1, int(self.fps * 0.8))
        rows = history.get("speed_history") or []
        before = [r for r in rows if int(r.get("frame", -1)) < event_frame]
        valid = []
        for row in before:
            if row.get("speed_status") not in ("calibrated_reference", "estimated_prior"):
                continue
            try:
                value = float(row.get("speed_kmh"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0.0:
                valid.append(row)
        return {
            "candidate_frame": event_frame,
            "track_observations": int(history.get("observations") or 0),
            "valid_metric_samples_before_candidate": len(valid),
            "valid_metric_samples_in_0_8s": sum(int(r.get("frame", -1)) >= start_08 for r in valid),
            "last_valid_metric_frame_before_candidate": max((int(r["frame"]) for r in valid), default=None),
            "metric_sources_before_candidate": sorted({r.get("speed_status") for r in valid}),
        }

    def detect(self, histories, tracked_objects=None):
        self.frame_index += 1
        if tracked_objects is None:
            return []

        current = {}
        for obj in tracked_objects:
            try:
                tid = int(obj["id"])
            except (KeyError, TypeError, ValueError):
                continue
            history = histories.get(tid)
            if (
                obj.get("class") == "car"
                and obj.get("bbox") is not None
                and history is not None
                and history.get("active", True)
                and history.get("class") == "car"
            ):
                current[tid] = obj

        ids = sorted(current)
        events = []
        seen_pairs = set()

        for index, id1 in enumerate(ids):
            h1 = histories[id1]
            for id2 in ids[index + 1:]:
                h2 = histories[id2]
                pair = (id1, id2)
                seen_pairs.add(pair)

                obs1 = int(h1.get("observations", len(h1.get("trajectory", []))))
                obs2 = int(h2.get("observations", len(h2.get("trajectory", []))))
                iou, gap, small_diag, location = self.geometry(
                    current[id1]["bbox"], current[id2]["bbox"]
                )
                state = self.pair_states.setdefault(pair, {
                    "near_frames": 0,
                    "last_near_frame": None,
                    "emitted": False,
                    "best_gap_ratio": float("inf"),
                    "best_iou": 0.0,
                })

                center1 = current[id1].get("center")
                center2 = current[id2].get("center")
                if center1 is None:
                    b = current[id1]["bbox"]
                    center1 = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                if center2 is None:
                    b = current[id2]["bbox"]
                    center2 = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

                vel1 = self._estimate_image_velocity(h1)
                vel2 = self._estimate_image_velocity(h2)
                t_cpa, cpa_px, closing = self._predict_closest_approach(
                    center1, center2, vel1[:2], vel2[:2], self.prediction_horizon_s
                )
                relative_speed = math.hypot(vel2[0] - vel1[0], vel2[1] - vel1[1])

                gap_ratio = gap / max(small_diag, 1.0)
                overlap = iou >= self.min_iou
                strong_overlap = iou >= self.strong_iou
                strong_proximity = gap_ratio <= self.strong_gap_ratio
                moderate_proximity = (
                    gap_ratio <= self.proximity_ratio
                    and (closing >= self.min_closing_px_s or
                         (t_cpa is not None and 0.0 <= t_cpa <= self.prediction_horizon_s
                          and cpa_px <= small_diag * self.predicted_cpa_ratio))
                )
                predicted_conflict = (
                    t_cpa is not None
                    and 0.0 <= t_cpa <= self.prediction_horizon_s
                    and cpa_px <= max(8.0, small_diag * self.predicted_cpa_ratio)
                    and gap_ratio <= min(0.55, self.proximity_ratio * 2.0)
                )

                near = overlap or strong_proximity or moderate_proximity or predicted_conflict
                if near:
                    if state["last_near_frame"] is None or self.frame_index - state["last_near_frame"] > 1:
                        state["near_frames"] = 1
                    else:
                        state["near_frames"] += 1
                    state["last_near_frame"] = self.frame_index
                elif state["last_near_frame"] is not None and self.frame_index - state["last_near_frame"] > self.reset_separation_frames:
                    state["near_frames"] = 0
                    state["last_near_frame"] = None

                state["best_gap_ratio"] = min(state["best_gap_ratio"], gap_ratio)
                state["best_iou"] = max(state["best_iou"], iou)

                enough_observations = obs1 >= self.min_observations and obs2 >= self.min_observations
                strong_contact_evidence = (
                    strong_overlap
                    or (
                        strong_proximity
                        and (
                            closing >= self.min_closing_px_s
                            or predicted_conflict
                            or (state["near_frames"] >= self.min_near_frames and relative_speed >= 5.0)
                        )
                    )
                )
                persistent_contact_evidence = (
                    state["near_frames"] >= self.min_near_frames
                    and (
                        closing >= self.min_closing_px_s
                        or predicted_conflict
                        or overlap
                        or relative_speed >= 5.0
                    )
                )
                confirmed_geometry = strong_contact_evidence or persistent_contact_evidence

                if not enough_observations:
                    reason = f"Observasi belum cukup (V1={obs1}, V2={obs2}; minimum={self.min_observations})."
                elif not near:
                    reason = "Bounding box belum cukup dekat; tidak ada overlap atau lintasan mendekat yang kuat."
                elif not confirmed_geometry:
                    reason = "Kedekatan hanya sesaat; menunggu bukti overlap, pendekatan, atau persistensi."
                elif state["emitted"]:
                    reason = "Pasangan ini sudah pernah dicatat sebagai kandidat."
                else:
                    reason = "Bukti kedekatan/overlap pasangan memenuhi ambang kandidat."

                self._update_diagnostic(
                    pair, obs1, obs2, iou, gap, small_diag, closing,
                    cpa_px if t_cpa is not None and 0 <= t_cpa <= self.prediction_horizon_s else None,
                    t_cpa if t_cpa is not None and 0 <= t_cpa <= self.prediction_horizon_s else None,
                    state["near_frames"], False, reason,
                )

                if enough_observations and confirmed_geometry and not state["emitted"]:
                    classification = self.classifier.classify(h1, h2, id1, id2)
                    role_v1 = int(classification.get("vehicle1_id", id1))
                    role_v2 = int(classification.get("vehicle2_id", id2))
                    if role_v1 not in current or role_v2 not in current:
                        role_v1, role_v2 = id1, id2

                    role_h1 = histories[role_v1]
                    role_h2 = histories[role_v2]
                    speed1 = float(role_h1.get("speed") or 0.0)
                    speed2 = float(role_h2.get("speed") or 0.0)
                    vx1, vy1 = float(role_h1.get("velocity_x") or 0.0), float(role_h1.get("velocity_y") or 0.0)
                    vx2, vy2 = float(role_h2.get("velocity_x") or 0.0), float(role_h2.get("velocity_y") or 0.0)
                    relative = math.hypot(vx1 - vx2, vy1 - vy2)
                    acceleration1 = float(role_h1.get("acceleration") or 0.0)
                    acceleration2 = float(role_h2.get("acceleration") or 0.0)

                    box_v1 = current[role_v1]["bbox"]
                    box_v2 = current[role_v2]["bbox"]
                    iou_role, gap_role, _, location_role = self.geometry(box_v1, box_v2)
                    role_diag = {
                        "bbox_overlap": bool(iou_role > 0),
                        "bbox_iou": round(iou_role, 5),
                        "bbox_gap_px": round(gap_role, 2),
                        "closing_speed_px_s": round(closing, 2),
                        "predicted_cpa_px": round(cpa_px, 2) if cpa_px is not None else None,
                        "predicted_cpa_time_s": round(t_cpa, 3) if t_cpa is not None else None,
                    }

                    events.append({
                        "v1": role_v1,
                        "v2": role_v2,
                        "pair_ids": [id1, id2],
                        "vehicle1_role": classification.get("vehicle1_role", "Kendaraan acuan"),
                        "vehicle2_role": classification.get("vehicle2_role", "Kendaraan pasangan"),
                        "role_assignment_reason": classification.get("role_assignment_reason", ""),
                        "class1": "car",
                        "class2": "car",
                        "bbox_v1": list(map(int, box_v1)),
                        "bbox_v2": list(map(int, box_v2)),
                        "location": location_role,
                        "bbox_iou": round(iou_role, 5),
                        "bbox_gap_px": round(gap_role, 2),
                        "closing_speed_px_s": round(closing, 2),
                        "relative_speed_px_s": round(relative, 2),
                        "speed_v1_px_s": round(speed1, 2),
                        "speed_v2_px_s": round(speed2, 2),
                        "acceleration_v1_px_s2": round(acceleration1, 2),
                        "acceleration_v2_px_s2": round(acceleration2, 2),
                        "heading_v1_deg": classification.get("heading_v1_deg", classification.get("heading_a_deg")),
                        "heading_v2_deg": classification.get("heading_v2_deg", classification.get("heading_b_deg")),
                        "approach_heading_v1_deg": classification.get("approach_heading_v1_deg"),
                        "approach_heading_v2_deg": classification.get("approach_heading_v2_deg"),
                        "motion_relation": classification.get("motion_relation", "Tidak diketahui"),
                        "crash_type_code": classification.get("crash_type_code"),
                        "crash_type": classification.get("crash_type", "Belum dapat diklasifikasikan dari lintasan yang tersedia"),
                        "impact_angle_deg": classification.get("approach_angle_deg"),
                        "impact_angle_reference": classification.get("impact_angle_reference", "image_plane"),
                        "classification_coordinate_space": classification.get("classification_coordinate_space", "image_px"),
                        "maneuver_v1": classification.get("maneuver_v1", classification.get("maneuver_a", "Lintasan belum cukup")),
                        "maneuver_v2": classification.get("maneuver_v2", classification.get("maneuver_b", "Lintasan belum cukup")),
                        "contact_side_v1": classification.get("contact_side_v1", "Tidak diketahui"),
                        "contact_side_v2": classification.get("contact_side_v2", "Tidak diketahui"),
                        "classification_confidence": classification.get("classification_confidence", "Rendah"),
                        "classification_evidence": classification.get("classification_evidence", []),
                        "rule_reason": classification.get("rule_reason", ""),
                        "event_status": "potential_contact",
                        "event_frame": self.frame_index,
                        "evidence": {
                            **role_diag,
                            "pair_near_frames": state["near_frames"],
                            "abrupt_deceleration_v1": acceleration1 < -1000.0,
                            "abrupt_deceleration_v2": acceleration2 < -1000.0,
                            "classification_confidence": classification.get("classification_confidence", "Rendah"),
                            "speed_readiness_at_initial_candidate": {
                                "v1": self._speed_readiness_at_event(role_h1),
                                "v2": self._speed_readiness_at_event(role_h2),
                            },
                            "detector_rule": (
                                "strong_bbox_overlap" if strong_overlap else
                                "strong_bbox_proximity" if strong_proximity else
                                "persistent_or_predicted_proximity"
                            ),
                        },
                    })
                    state["emitted"] = True
                    self._update_diagnostic(
                        pair, obs1, obs2, iou, gap, small_diag, closing,
                        cpa_px if t_cpa is not None and 0 <= t_cpa <= self.prediction_horizon_s else None,
                        t_cpa if t_cpa is not None and 0 <= t_cpa <= self.prediction_horizon_s else None,
                        state["near_frames"], True,
                        "Kandidat kontak dicatat; jenis kejadian ditentukan terpisah oleh classifier."
                    )

        # Re-arm only after the pair has disappeared/separated long enough.
        for pair, state in list(self.pair_states.items()):
            if pair not in seen_pairs and state["last_near_frame"] is not None:
                if self.frame_index - state["last_near_frame"] > self.reset_separation_frames:
                    state.update({"near_frames": 0, "last_near_frame": None, "emitted": False})

        return events

    def _truncate_history(self, history, end_frame):
        """Copy a track history and retain only observations up to a candidate frame."""
        import copy
        out = copy.copy(history)
        frames = list(history.get("frame_indices") or [])
        keep = [i for i, frame in enumerate(frames) if int(frame) <= int(end_frame)]
        aligned_keys = (
            "trajectory", "footpoint_trajectory", "analysis_trajectory",
            "world_trajectory", "bbox_history", "frame_indices",
        )
        for key in aligned_keys:
            values = list(history.get(key) or [])
            if len(values) == len(frames):
                out[key] = [values[i] for i in keep]
        out["speed_history"] = [
            row for row in (history.get("speed_history") or [])
            if int(row.get("frame", -1)) <= int(end_frame)
        ]
        out["observations"] = len(out.get("frame_indices") or [])
        # Recompute image velocity from the truncated pre-contact track so a
        # post-impact turn/deceleration cannot leak into the pre-impact matrix.
        vx, vy, _ = self._estimate_image_velocity(out)
        out["velocity_x"] = vx
        out["velocity_y"] = vy
        if out["speed_history"]:
            last_speed = out["speed_history"][-1]
            out["speed"] = last_speed.get("speed_px_s")
            out["speed_kmh"] = last_speed.get("speed_kmh")
            out["speed_status"] = last_speed.get("speed_status", "uncalibrated")
            out["speed_method"] = last_speed.get("speed_method", "image_motion_only")
            out["velocity_m_s"] = last_speed.get("velocity_m_s")
        return out

    def finalize(self, histories):
        """Post-video synchronized-track pass to recover missed brief contacts.

        This is intentionally a candidate screen, not proof of physical impact.
        It compares bounding boxes only at frames where both track IDs were
        observed, then classifies using each trajectory truncated at that frame.
        """
        events = []
        ids = sorted(int(k) for k in (histories or {}).keys())
        for index, id1 in enumerate(ids):
            h1 = histories.get(id1)
            if not h1 or h1.get("class") != "car":
                continue
            frames1 = list(h1.get("frame_indices") or [])
            boxes1 = list(h1.get("bbox_history") or [])
            centres1 = list(h1.get("trajectory") or [])
            if not frames1 or len(frames1) != len(boxes1):
                continue
            map1 = {int(f): (boxes1[i], centres1[i] if i < len(centres1) else None)
                    for i, f in enumerate(frames1)}

            for id2 in ids[index + 1:]:
                h2 = histories.get(id2)
                if not h2 or h2.get("class") != "car":
                    continue
                pair = (id1, id2)
                state = self.pair_states.setdefault(pair, {
                    "near_frames": 0, "last_near_frame": None, "emitted": False,
                    "best_gap_ratio": float("inf"), "best_iou": 0.0,
                })
                if state.get("emitted"):
                    continue
                frames2 = list(h2.get("frame_indices") or [])
                boxes2 = list(h2.get("bbox_history") or [])
                centres2 = list(h2.get("trajectory") or [])
                if not frames2 or len(frames2) != len(boxes2):
                    continue
                map2 = {int(f): (boxes2[i], centres2[i] if i < len(centres2) else None)
                        for i, f in enumerate(frames2)}
                common = sorted(set(map1).intersection(map2))
                if not common:
                    d = self._diagnostic_record(pair)
                    d["last_reason"] = "Kedua ID tidak memiliki observasi pada frame yang sama."
                    d["frames_compared"] = 0
                    continue

                best = None
                first_overlap = None
                max_iou = 0.0
                min_gap = float("inf")
                min_ratio = float("inf")
                for frame in common:
                    box1, center1 = map1[frame]
                    box2, center2 = map2[frame]
                    iou, gap, small_diag, location = self.geometry(box1, box2)
                    ratio = gap / max(small_diag, 1.0)
                    max_iou = max(max_iou, iou)
                    if gap < min_gap:
                        min_gap = gap
                    min_ratio = min(min_ratio, ratio)
                    score = (iou, -ratio)
                    row = {"frame": frame, "bbox1": box1, "bbox2": box2,
                           "iou": iou, "gap": gap, "diag": small_diag,
                           "ratio": ratio, "location": location, "score": score}
                    if best is None or score > best["score"]:
                        best = row
                    if first_overlap is None and iou >= 0.0005:
                        first_overlap = row

                obs1 = int(h1.get("observations", len(frames1)))
                obs2 = int(h2.get("observations", len(frames2)))
                d = self._diagnostic_record(pair)
                d["frames_compared"] = max(int(d.get("frames_compared", 0)), len(common))
                d["min_observations_v1"] = obs1
                d["min_observations_v2"] = obs2
                d["min_bbox_gap_px"] = round(min_gap, 2)
                d["min_normalized_gap"] = round(min_ratio, 4)
                d["max_bbox_iou"] = round(max_iou, 5)
                d["last_frame"] = int(best["frame"]) if best else None

                overlap_evidence = max_iou >= 0.0005
                close_evidence = min_ratio <= 0.14
                chosen = first_overlap if first_overlap is not None else best
                if obs1 < 3 or obs2 < 3:
                    d["last_reason"] = f"Lintasan terlalu pendek untuk konfirmasi (V1={obs1}, V2={obs2}; minimum=3)."
                    continue
                if not (overlap_evidence or close_evidence):
                    d["last_reason"] = (
                        "Pemeriksaan akhir: tidak ada overlap bbox dan jarak minimum "
                        f"masih di atas ambang (rasio={min_ratio:.3f})."
                    )
                    continue

                # Use the frame with the strongest overlap; otherwise the closest
                # synchronized frame. Never use future/post-contact samples.
                event_frame = int(chosen["frame"])
                pre1 = self._truncate_history(h1, event_frame)
                pre2 = self._truncate_history(h2, event_frame)
                classification = self.classifier.classify(pre1, pre2, id1, id2)
                role_v1 = int(classification.get("vehicle1_id", id1))
                role_v2 = int(classification.get("vehicle2_id", id2))
                if role_v1 not in (id1, id2) or role_v2 not in (id1, id2) or role_v1 == role_v2:
                    role_v1, role_v2 = id1, id2
                role_h1 = pre1 if role_v1 == id1 else pre2
                role_h2 = pre2 if role_v2 == id2 else pre1
                box_v1 = chosen["bbox1"] if role_v1 == id1 else chosen["bbox2"]
                box_v2 = chosen["bbox2"] if role_v2 == id2 else chosen["bbox1"]
                iou_role, gap_role, _, location_role = self.geometry(box_v1, box_v2)
                vx1 = float(role_h1.get("velocity_x") or 0.0)
                vy1 = float(role_h1.get("velocity_y") or 0.0)
                vx2 = float(role_h2.get("velocity_x") or 0.0)
                vy2 = float(role_h2.get("velocity_y") or 0.0)
                relative_px = math.hypot(vx1 - vx2, vy1 - vy2)
                detector_rule = "posthoc_synchronized_bbox_overlap" if overlap_evidence else "posthoc_synchronized_close_bbox"
                event = {
                    "v1": role_v1, "v2": role_v2, "pair_ids": [id1, id2],
                    "vehicle1_role": classification.get("vehicle1_role", "Kendaraan acuan"),
                    "vehicle2_role": classification.get("vehicle2_role", "Kendaraan pasangan"),
                    "role_assignment_reason": classification.get("role_assignment_reason", ""),
                    "class1": "car", "class2": "car",
                    "bbox_v1": list(map(int, box_v1)), "bbox_v2": list(map(int, box_v2)),
                    "location": location_role, "bbox_iou": round(iou_role, 5),
                    "bbox_gap_px": round(gap_role, 2),
                    "closing_speed_px_s": classification.get("closing_speed_px_s"),
                    "relative_speed_px_s": round(relative_px, 2),
                    "speed_v1_px_s": round(float(role_h1.get("speed") or 0.0), 2),
                    "speed_v2_px_s": round(float(role_h2.get("speed") or 0.0), 2),
                    "speed_v1_kmh": role_h1.get("speed_kmh"),
                    "speed_v2_kmh": role_h2.get("speed_kmh"),
                    "speed_v1_status": role_h1.get("speed_status", "uncalibrated"),
                    "speed_v2_status": role_h2.get("speed_status", "uncalibrated"),
                    "heading_v1_deg": classification.get("heading_v1_deg", classification.get("heading_a_deg")),
                    "heading_v2_deg": classification.get("heading_v2_deg", classification.get("heading_b_deg")),
                    "approach_heading_v1_deg": classification.get("approach_heading_v1_deg"),
                    "approach_heading_v2_deg": classification.get("approach_heading_v2_deg"),
                    "motion_relation": classification.get("motion_relation", "Tidak diketahui"),
                    "crash_type_code": classification.get("crash_type_code"),
                    "crash_type": classification.get("crash_type", "Belum dapat diklasifikasikan dari lintasan yang tersedia"),
                    "impact_angle_deg": classification.get("approach_angle_deg"),
                    "impact_angle_reference": classification.get("impact_angle_reference", "image_plane"),
                    "classification_coordinate_space": classification.get("classification_coordinate_space", "image_px"),
                    "maneuver_v1": classification.get("maneuver_v1", classification.get("maneuver_a", "Lintasan belum cukup")),
                    "maneuver_v2": classification.get("maneuver_v2", classification.get("maneuver_b", "Lintasan belum cukup")),
                    "contact_side_v1": classification.get("contact_side_v1", "Tidak diketahui"),
                    "contact_side_v2": classification.get("contact_side_v2", "Tidak diketahui"),
                    "classification_confidence": classification.get("classification_confidence", "Rendah"),
                    "classification_evidence": classification.get("classification_evidence", []),
                    "rule_reason": classification.get("rule_reason", ""),
                    "event_status": "potential_contact", "event_frame": event_frame,
                    "evidence": {
                        "detector_rule": detector_rule, "common_observation_frames": len(common),
                        "max_bbox_iou": round(max_iou, 5), "event_frame_bbox_iou": round(chosen["iou"], 5),
                        "min_bbox_gap_px": round(min_gap, 2),
                        "min_normalized_gap": round(min_ratio, 4),
                        "candidate_is_not_physical_proof": True,
                    },
                }
                events.append(event)
                state["emitted"] = True
                d["candidate_emitted"] = True
                d["last_reason"] = "Kandidat ditemukan pada pemeriksaan akhir lintasan tersinkron; bukan bukti final benturan fisik."
        return events

    def get_diagnostics(self):
        """Return per-pair evidence summary, sorted by closest normalized bbox gap."""
        rows = [dict(value) for value in self.pair_diagnostics.values()]
        rows.sort(key=lambda d: (
            float("inf") if d.get("min_normalized_gap") is None else d["min_normalized_gap"],
            -float(d.get("max_bbox_iou") or 0.0),
        ))
        return rows

    def diagnostics_message(self):
        rows = self.get_diagnostics()
        if not rows:
            return "Tidak ada pasangan ID mobil yang aktif bersamaan untuk dibandingkan."
        d = rows[0]
        ids = d.get("vehicle_ids", ["?", "?"])
        return (
            f"Pasangan terdekat ID {ids[0]}-{ids[1]}: "
            f"jarak bbox minimum {d.get('min_bbox_gap_px')} px "
            f"(rasio {d.get('min_normalized_gap')}), "
            f"IoU maksimum {d.get('max_bbox_iou')}, "
            f"closing maksimum {d.get('max_closing_speed_px_s')} px/s. "
            f"Diagnosis: {d.get('last_reason')}"
        )
