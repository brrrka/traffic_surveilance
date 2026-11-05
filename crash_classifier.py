"""Rule-based classifier for the seven ANCHOR VI Car-Car scenarios.

The classifier prefers per-camera ground-plane trajectories when a valid
homography is available. Otherwise it falls back to image-plane motion and
marks the classification as low-confidence because perspective can distort
angles and lateral displacement. A lane-change/turn distinction remains a
rule-based heuristic unless lane boundaries are also available.
"""
from __future__ import annotations

import math
from statistics import median

SCENARIOS = {
    "REAR_MOVING": "Tabrakan dari Belakang (Rear Moving)",
    "TURN_ACROSS_PATH_ONCOMING": "Berbelok Memotong Jalur Kendaraan dari Arah Berlawanan (Turn Across Path – Oncoming)",
    "TURN_ACROSS_PATH_LATERAL": "Berbelok Memotong Jalur Kendaraan dari Samping (Turn Across Path – Lateral)",
    "CROSSING": "Persilangan Jalur Tegak Lurus (Crossing)",
    "SAME_DIRECTION_TURN": "Dua Kendaraan Berbelok ke Arah yang Sama (Same Direction Turn)",
    "CHANGE_LANE_ONCOMING": "Pindah Jalur ke Jalur Kendaraan Berlawanan Arah (Change Lane – Oncoming)",
    "CHANGE_LANE_OVERTAKING": "Pindah Jalur untuk Mendahului (Change Lane – Overtaking)",
}
UNCLASSIFIED = "Belum dapat diklasifikasikan dari lintasan yang tersedia"


def angle_difference(a, b):
    d = abs((float(a) - float(b)) % 360.0)
    return min(d, 360.0 - d)


def signed_angle_difference(a, b):
    """Signed shortest rotation from heading a to heading b (-180..180)."""
    return (float(b) - float(a) + 180.0) % 360.0 - 180.0


def _mean_heading(headings):
    if not headings:
        return None
    sx = sum(math.cos(math.radians(h)) for h in headings)
    sy = sum(math.sin(math.radians(h)) for h in headings)
    if math.hypot(sx, sy) < 1e-8:
        return None
    return math.degrees(math.atan2(sy, sx)) % 360.0


def _trajectory_data(vehicle):
    # Use metric ground-plane points whenever this CCTV has a valid per-camera
    # homography. Fall back to image footpoints only when metric coordinates do
    # not exist. This keeps scenario rules in one coordinate system per track.
    points = vehicle.get("analysis_trajectory") or vehicle.get("world_trajectory") or vehicle.get("trajectory", [])
    frames = vehicle.get("frame_indices", [])
    if len(points) != len(frames):
        frames = list(range(len(points)))
    rows = []
    for frame, point in zip(frames, points):
        if point is None or len(point) < 2:
            continue
        rows.append((int(frame), float(point[0]), float(point[1])))
    return rows


class RuleBasedCrashClassifier:
    """Classifies Car-Car candidate events into the seven PPT scenarios."""

    def __init__(self, min_travel_px=8.0, turn_threshold_deg=25.0,
                 lane_shift_ratio=0.12, same_direction_tolerance=35.0):
        self.min_travel_px = float(min_travel_px)
        self.turn_threshold_deg = float(turn_threshold_deg)
        self.lane_shift_ratio = float(lane_shift_ratio)
        self.same_direction_tolerance = float(same_direction_tolerance)

    def _features(self, vehicle):
        data = _trajectory_data(vehicle)
        coordinate_space = vehicle.get("analysis_coordinate_space", "image_px")
        metric_space = coordinate_space == "ground_plane_m"
        min_travel = 2.5 if metric_space else self.min_travel_px
        min_segment = 0.025 if metric_space else 1.0
        min_lateral = 0.65 if metric_space else 8.0
        min_longitudinal = 1.5 if metric_space else 12.0
        if len(data) < 7:
            return {"valid": False, "maneuver": "Lintasan belum cukup", "turn_delta": None,
                    "coordinate_space": coordinate_space}

        # Use a recent window so old track history does not dominate a new maneuver.
        data = data[-24:]
        segments = []
        path_length = 0.0
        for a, b in zip(data[:-1], data[1:]):
            df = b[0] - a[0]
            if df <= 0 or df > 4:
                continue
            dx, dy = b[1] - a[1], b[2] - a[2]
            dist = math.hypot(dx, dy)
            if dist < min_segment:
                continue
            segments.append((b[0], math.degrees(math.atan2(dy, dx)) % 360.0, dx, dy, dist))
            path_length += dist

        if len(segments) < 5 or path_length < min_travel:
            return {"valid": False, "maneuver": "Lintasan belum cukup", "turn_delta": None,
                    "coordinate_space": coordinate_space}

        n = len(segments)
        chunk = max(2, n // 5)
        start_heading = _mean_heading([s[1] for s in segments[:chunk]])
        end_heading = _mean_heading([s[1] for s in segments[-chunk:]])
        if start_heading is None or end_heading is None:
            return {"valid": False, "maneuver": "Lintasan belum cukup", "turn_delta": None,
                    "coordinate_space": coordinate_space}

        turn_delta = signed_angle_difference(start_heading, end_heading)
        x0, y0 = data[0][1], data[0][2]
        x1, y1 = data[-1][1], data[-1][2]
        dx, dy = x1 - x0, y1 - y0
        longitudinal = dx * math.cos(math.radians(start_heading)) + dy * math.sin(math.radians(start_heading))
        lateral = -dx * math.sin(math.radians(start_heading)) + dy * math.cos(math.radians(start_heading))
        displacement = math.hypot(dx, dy)
        curvature_ratio = path_length / max(displacement, 1e-6)

        if abs(turn_delta) >= self.turn_threshold_deg:
            maneuver = "Berbelok"
            maneuver_kind = "turn"
        elif (abs(lateral) >= max(min_lateral, abs(longitudinal) * self.lane_shift_ratio)
              and abs(longitudinal) >= min_longitudinal and abs(turn_delta) <= 22.0):
            maneuver = "Berpindah jalur (indikasi dari lintasan)"
            maneuver_kind = "lane_change"
        else:
            maneuver = "Bergerak lurus"
            maneuver_kind = "straight"

        # Initial and final positions/headings are retained for role assignment.
        first_frame = data[0][0]
        last_frame = data[-1][0]
        return {
            "valid": True,
            "maneuver": maneuver,
            "maneuver_kind": maneuver_kind,
            "turn_delta": turn_delta,
            "start_heading": start_heading,
            "end_heading": end_heading,
            "path_length": path_length,
            "longitudinal_displacement": longitudinal,
            "lateral_displacement": lateral,
            "curvature_ratio": curvature_ratio,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "start_point": (x0, y0),
            "end_point": (x1, y1),
            "data": data,
            "coordinate_space": coordinate_space,
        }

    def classify_maneuver(self, vehicle):
        return self._features(vehicle)["maneuver"]

    @staticmethod
    def closing_speed_px_s(v1, v2):
        p1 = v1.get("trajectory", [])
        p2 = v2.get("trajectory", [])
        if not p1 or not p2:
            return None
        rx = float(p2[-1][0]) - float(p1[-1][0])
        ry = float(p2[-1][1]) - float(p1[-1][1])
        distance = math.hypot(rx, ry)
        if distance <= 1e-6:
            return 0.0
        rvx = float(v2.get("velocity_x", 0.0)) - float(v1.get("velocity_x", 0.0))
        rvy = float(v2.get("velocity_y", 0.0)) - float(v1.get("velocity_y", 0.0))
        return max(0.0, -(rx * rvx + ry * rvy) / distance)

    @staticmethod
    def _closing_speed_analysis(v1, v2):
        """Closing speed in the shared analysis coordinate space (px/s or m/s)."""
        p1 = _trajectory_data(v1)
        p2 = _trajectory_data(v2)
        if not p1 or not p2:
            return 0.0
        x1, y1 = p1[-1][1], p1[-1][2]
        x2, y2 = p2[-1][1], p2[-1][2]
        rx, ry = x2 - x1, y2 - y1
        distance = math.hypot(rx, ry)
        if distance <= 1e-9:
            return 0.0
        metric = v1.get("analysis_coordinate_space") == v2.get("analysis_coordinate_space") == "ground_plane_m"
        if metric:
            vx1, vy1 = v1.get("world_velocity_x"), v1.get("world_velocity_y")
            vx2, vy2 = v2.get("world_velocity_x"), v2.get("world_velocity_y")
        else:
            vx1, vy1 = v1.get("velocity_x", 0.0), v1.get("velocity_y", 0.0)
            vx2, vy2 = v2.get("velocity_x", 0.0), v2.get("velocity_y", 0.0)
        if None in (vx1, vy1, vx2, vy2):
            return 0.0
        return max(0.0, -((vx2 - vx1) * rx + (vy2 - vy1) * ry) / distance)

    @staticmethod
    def _contact_side(vehicle, other_vehicle, heading=None):
        points1 = vehicle.get("analysis_trajectory") or vehicle.get("world_trajectory") or vehicle.get("trajectory", [])
        points2 = other_vehicle.get("analysis_trajectory") or other_vehicle.get("world_trajectory") or other_vehicle.get("trajectory", [])
        points1 = [p for p in points1 if p is not None]
        points2 = [p for p in points2 if p is not None]
        if not points1 or not points2:
            return "Tidak diketahui"
        if heading is None:
            heading = vehicle.get("analysis_heading", vehicle.get("heading"))
        if heading is None:
            return "Tidak diketahui"
        x1, y1 = points1[-1]
        x2, y2 = points2[-1]
        bearing = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0
        relative = signed_angle_difference(float(heading), bearing)
        if abs(relative) <= 45:
            return "Depan"
        if abs(relative) >= 135:
            return "Belakang"
        return "Sisi kanan" if relative > 0 else "Sisi kiri"

    @staticmethod
    def _position_projection(vehicle_a, vehicle_b, heading_deg, use_start=False):
        da, db = _trajectory_data(vehicle_a), _trajectory_data(vehicle_b)
        if not da or not db:
            return None
        common_start = max(da[0][0], db[0][0])
        common_end = min(da[-1][0], db[-1][0])
        if common_end <= common_start:
            return None
        frame = common_start if use_start else common_end

        def point_near(data, target):
            return min(data, key=lambda row: abs(row[0] - target))[1:3]

        pa = point_near(da, frame)
        pb = point_near(db, frame)
        dx, dy = pa[0] - pb[0], pa[1] - pb[1]
        return dx * math.cos(math.radians(heading_deg)) + dy * math.sin(math.radians(heading_deg))

    @staticmethod
    def _lateral_alignment(v1, v2, heading_deg):
        p1 = [p for p in (v1.get("analysis_trajectory") or v1.get("world_trajectory") or v1.get("trajectory", [])) if p is not None]
        p2 = [p for p in (v2.get("analysis_trajectory") or v2.get("world_trajectory") or v2.get("trajectory", [])) if p is not None]
        if not p1 or not p2:
            return None
        dx = float(p2[-1][0]) - float(p1[-1][0])
        dy = float(p2[-1][1]) - float(p1[-1][1])
        return abs(-dx * math.sin(math.radians(heading_deg)) + dy * math.cos(math.radians(heading_deg)))

    @staticmethod
    def _id(vehicle, fallback):
        try:
            return int(vehicle.get("track_id", fallback))
        except (TypeError, ValueError):
            return int(fallback)

    def classify(self, vehicle_a, vehicle_b, id_a=None, id_b=None):
        """Return classification plus role-ordered Vehicle 1 and Vehicle 2 IDs."""
        id_a = self._id(vehicle_a, id_a if id_a is not None else 1)
        id_b = self._id(vehicle_b, id_b if id_b is not None else 2)
        fa, fb = self._features(vehicle_a), self._features(vehicle_b)
        result = {
            "crash_type_code": None,
            "crash_type": UNCLASSIFIED,
            "maneuver_a": fa["maneuver"],
            "maneuver_b": fb["maneuver"],
            "maneuver_v1": fa["maneuver"],
            "maneuver_v2": fb["maneuver"],
            "heading_a_deg": round(fa["end_heading"], 1) if fa.get("valid") else None,
            "heading_b_deg": round(fb["end_heading"], 1) if fb.get("valid") else None,
            "approach_angle_deg": None,
            "motion_relation": "Tidak diketahui",
            "vehicle1_id": id_a,
            "vehicle2_id": id_b,
            "vehicle1_role": "Kendaraan acuan",
            "vehicle2_role": "Kendaraan pasangan",
            "role_assignment_reason": "Belum ada bukti cukup untuk menetapkan peran khusus V1/V2.",
            "contact_side_v1": "Tidak diketahui",
            "contact_side_v2": "Tidak diketahui",
            "closing_speed_px_s": self.closing_speed_px_s(vehicle_a, vehicle_b),
            "classification_coordinate_space": ("ground_plane_m" if fa.get("coordinate_space") == "ground_plane_m" and fb.get("coordinate_space") == "ground_plane_m" else "image_px"),
            "impact_angle_reference": ("ground_plane" if fa.get("coordinate_space") == "ground_plane_m" and fb.get("coordinate_space") == "ground_plane_m" else "image_plane"),
            "classification_confidence": "Rendah",
            "classification_evidence": [],
            "rule_reason": "Lintasan belum cukup untuk mencocokkan salah satu dari tujuh skenario ANCHOR VI.",
        }
        if not fa.get("valid") or not fb.get("valid"):
            return result

        # Variable 9 is the angle between the two pre-impact approach headings.
        # Keep the later heading comparison separate; it may already include
        # post-manoeuvre or contact-induced rotation.
        approach_angle = angle_difference(fa["start_heading"], fb["start_heading"])
        relative_heading_at_contact = angle_difference(fa["end_heading"], fb["end_heading"])
        result["approach_angle_deg"] = round(approach_angle, 1)
        result["relative_heading_at_contact_deg"] = round(relative_heading_at_contact, 1)
        result["classification_coordinate_space"] = "ground_plane_m" if fa.get("coordinate_space") == fb.get("coordinate_space") == "ground_plane_m" else "image_px"
        result["impact_angle_reference"] = "ground_plane" if result["classification_coordinate_space"] == "ground_plane_m" else "image_plane"
        if approach_angle <= 35:
            relation = "Searah"
        elif approach_angle >= 145:
            relation = "Berlawanan arah"
        elif 55 <= approach_angle <= 125:
            relation = "Tegak lurus / lateral"
        else:
            relation = "Miring / tidak tegas"
        result["motion_relation"] = relation

        kind_a, kind_b = fa["maneuver_kind"], fb["maneuver_kind"]
        code = None
        v1_id, v2_id = id_a, id_b
        v1, v2 = vehicle_a, vehicle_b
        f1, f2 = fa, fb
        role1 = "Kendaraan acuan"
        role2 = "Kendaraan pasangan"
        role_reason = "Penetapan peran berdasarkan aturan skenario."
        evidence = []
        reason = ""

        # Priority 1: lane-change patterns.
        if relation == "Berlawanan arah" and "lane_change" in (kind_a, kind_b):
            actor_is_a = kind_a == "lane_change"
            code = "CHANGE_LANE_ONCOMING"
            if not actor_is_a:
                v1_id, v2_id, v1, v2, f1, f2 = id_b, id_a, vehicle_b, vehicle_a, fb, fa
            role1, role2 = "Mobil yang berpindah jalur", "Mobil dari arah berlawanan"
            role_reason = "V1 ditetapkan sebagai mobil yang menunjukkan indikasi berpindah ke jalur arus berlawanan."
            evidence = ["Arah pendekatan kedua kendaraan berlawanan", "Salah satu lintasan menunjukkan perpindahan lateral dengan arah akhir mendekati arah semula"]
            reason = "Pola perpindahan jalur terindikasi menuju lintasan kendaraan dari arah berlawanan."

        elif relation == "Searah" and "lane_change" in (kind_a, kind_b):
            # Overtaking requires evidence that the lateral-moving vehicle changes
            # longitudinal order relative to the other vehicle.
            actor_is_a = kind_a == "lane_change"
            actor, other = (vehicle_a, vehicle_b) if actor_is_a else (vehicle_b, vehicle_a)
            actor_id, other_id = (id_a, id_b) if actor_is_a else (id_b, id_a)
            actor_features = fa if actor_is_a else fb
            shared_heading = _mean_heading([fa["start_heading"], fb["start_heading"]])
            start_order = self._position_projection(actor, other, shared_heading, use_start=True)
            end_order = self._position_projection(actor, other, shared_heading, use_start=False)
            if start_order is not None and end_order is not None and start_order < -2 and end_order > 2:
                code = "CHANGE_LANE_OVERTAKING"
                v1_id, v2_id, v1, v2 = actor_id, other_id, actor, other
                f1, f2 = actor_features, (fb if actor_is_a else fa)
                role1, role2 = "Mobil yang mendahului", "Mobil yang didahului"
                role_reason = "V1 ditetapkan sebagai mobil yang berpindah jalur dan berubah posisi dari belakang menjadi di depan V2."
                evidence = ["Arah kedua kendaraan searah", "Terdapat indikasi perpindahan lateral", "Urutan posisi longitudinal berubah dari belakang menjadi di depan"]
                reason = "Lintasan menunjukkan pola perpindahan jalur disertai gerakan mendahului."

        # Priority 2: turning across path.
        if code is None and relation == "Berlawanan arah" and "turn" in (kind_a, kind_b):
            actor_is_a = kind_a == "turn"
            code = "TURN_ACROSS_PATH_ONCOMING"
            if not actor_is_a:
                v1_id, v2_id, v1, v2, f1, f2 = id_b, id_a, vehicle_b, vehicle_a, fb, fa
            role1, role2 = "Mobil yang berbelok", "Mobil dari arah berlawanan"
            role_reason = "V1 ditetapkan sebagai mobil yang lintasannya menunjukkan manuver berbelok memotong arus berlawanan."
            evidence = ["Arah pendekatan kedua kendaraan berlawanan", "Salah satu kendaraan menunjukkan perubahan arah lintasan yang kuat"]
            reason = "Manuver berbelok terindikasi memotong lintasan kendaraan dari arah berlawanan."

        elif code is None and relation == "Tegak lurus / lateral" and "turn" in (kind_a, kind_b):
            actor_is_a = kind_a == "turn"
            code = "TURN_ACROSS_PATH_LATERAL"
            if not actor_is_a:
                v1_id, v2_id, v1, v2, f1, f2 = id_b, id_a, vehicle_b, vehicle_a, fb, fa
            role1, role2 = "Mobil yang berbelok", "Mobil dari arah samping"
            role_reason = "V1 ditetapkan sebagai mobil yang berbelok; V2 merupakan kendaraan yang mendekat dari arah lateral."
            evidence = ["Arah pendekatan membentuk sudut lateral", "Salah satu kendaraan menunjukkan manuver berbelok"]
            reason = "Manuver berbelok terindikasi memotong lintasan kendaraan yang datang dari samping."

        # Priority 3: two vehicles turning in the same direction.
        if code is None and relation == "Searah" and kind_a == "turn" and kind_b == "turn":
            same_turn_sign = fa["turn_delta"] * fb["turn_delta"] > 0
            if same_turn_sign:
                code = "SAME_DIRECTION_TURN"
                # First vehicle to begin its turn is V1; if tied, use stable lower ID.
                # Track onset is not explicitly available; use lower ID as stable tie-break.
                if id_b < id_a:
                    v1_id, v2_id, v1, v2, f1, f2 = id_b, id_a, vehicle_b, vehicle_a, fb, fa
                role1, role2 = "Mobil acuan pada manuver belok", "Mobil pasangan pada manuver belok"
                role_reason = "Kedua kendaraan menunjukkan perubahan arah belok dengan tanda yang sama; ID tracking lebih kecil dipakai sebagai V1 karena urutan awal belok tidak dapat dipastikan dari data ini."
                evidence = ["Arah pendekatan kedua kendaraan searah", "Kedua lintasan menunjukkan perubahan arah belok ke sisi gambar yang sama"]
                reason = "Kedua kendaraan menunjukkan pola belok searah."

        # Priority 4: rear-moving conflict.
        if code is None and relation == "Searah" and kind_a == "straight" and kind_b == "straight":
            shared_heading = _mean_heading([fa["end_heading"], fb["end_heading"]])
            lateral_gap = self._lateral_alignment(vehicle_a, vehicle_b, shared_heading)
            metric_space = fa.get("coordinate_space") == fb.get("coordinate_space") == "ground_plane_m"
            small_diag = min(float(vehicle_a.get("bbox_history", [[0, 0, 40, 40]])[-1][2] - vehicle_a.get("bbox_history", [[0, 0, 40, 40]])[-1][0]),
                             float(vehicle_b.get("bbox_history", [[0, 0, 40, 40]])[-1][2] - vehicle_b.get("bbox_history", [[0, 0, 40, 40]])[-1][0]))
            lateral_limit = 1.5 if metric_space else max(20.0, small_diag * 0.55)
            closing = self._closing_speed_analysis(vehicle_a, vehicle_b)
            if (lateral_gap is not None
                and lateral_gap <= lateral_limit
                and closing > (0.15 if metric_space else 0.1)):
                code = "REAR_MOVING"
                # The rear vehicle is the one with the other vehicle ahead along travel direction.
                pa = _trajectory_data(vehicle_a)[-1][1:3]
                pb = _trajectory_data(vehicle_b)[-1][1:3]
                dx, dy = float(pb[0]) - float(pa[0]), float(pb[1]) - float(pa[1])
                longitudinal_b_from_a = dx * math.cos(math.radians(shared_heading)) + dy * math.sin(math.radians(shared_heading))
                if longitudinal_b_from_a >= 0:
                    v1_id, v2_id, v1, v2, f1, f2 = id_a, id_b, vehicle_a, vehicle_b, fa, fb
                else:
                    v1_id, v2_id, v1, v2, f1, f2 = id_b, id_a, vehicle_b, vehicle_a, fb, fa
                role1, role2 = "Mobil belakang", "Mobil depan"
                role_reason = "V1 ditetapkan sebagai mobil yang berada di belakang; V2 sebagai mobil yang berada di depan pada arah gerak yang sama."
                evidence = ["Arah gerak kedua kendaraan searah", "Kedua kendaraan relatif sejajar pada bidang gambar", "Posisi longitudinal menunjukkan satu mobil berada di belakang mobil lainnya"]
                reason = "Pola dua kendaraan bergerak searah dan relatif segaris; V1 berada di belakang V2."

        # Priority 5: perpendicular crossing with both vehicles near-straight.
        if code is None and relation == "Tegak lurus / lateral" and kind_a == "straight" and kind_b == "straight":
            code = "CROSSING"
            # Crossing has no unique initiating actor in the PPT; use stable ID order.
            if id_b < id_a:
                v1_id, v2_id, v1, v2, f1, f2 = id_b, id_a, vehicle_b, vehicle_a, fb, fa
            role1, role2 = "Kendaraan acuan pada persilangan", "Kendaraan dari arah tegak lurus"
            role_reason = "Tidak ada aktor tunggal yang dapat ditentukan dari lintasan; V1/V2 ditetapkan menurut ID tracking untuk konsistensi pencatatan."
            evidence = ["Arah gerak kedua kendaraan mendekati tegak lurus", "Kedua kendaraan tidak menunjukkan manuver belok yang cukup kuat"]
            reason = "Dua lintasan kendaraan saling melintasi dari arah yang hampir tegak lurus."

        if code is None:
            result.update({
                "approach_angle_deg": round(approach_angle, 1),
                "maneuver_v1": fa["maneuver"],
                "maneuver_v2": fb["maneuver"],
                "rule_reason": "Pola lintasan tidak memenuhi aturan yang cukup kuat untuk salah satu dari tujuh skenario ANCHOR VI.",
                "classification_evidence": [f"Hubungan arah: {relation}", f"Manuver kendaraan A: {fa['maneuver']}", f"Manuver kendaraan B: {fb['maneuver']}"],
            })
            return result

        result.update({
            "crash_type_code": code,
            "crash_type": SCENARIOS[code],
            "vehicle1_id": v1_id,
            "vehicle2_id": v2_id,
            "vehicle1_role": role1,
            "vehicle2_role": role2,
            "role_assignment_reason": role_reason,
            "maneuver_v1": f1["maneuver"],
            "maneuver_v2": f2["maneuver"],
            "approach_heading_v1_deg": round(f1["start_heading"], 1),
            "approach_heading_v2_deg": round(f2["start_heading"], 1),
            "heading_v1_deg": round(f1["end_heading"], 1),
            "heading_v2_deg": round(f2["end_heading"], 1),
            "approach_angle_deg": round(approach_angle, 1),
            "contact_side_v1": self._contact_side(v1, v2, f1["end_heading"]),
            "contact_side_v2": self._contact_side(v2, v1, f2["end_heading"]),
            "classification_confidence": ("Sedang" if len(evidence) >= 3 else "Rendah") if result.get("classification_coordinate_space") == "ground_plane_m" else "Rendah",
            "classification_evidence": evidence + ([] if result.get("classification_coordinate_space") == "ground_plane_m" else ["Klasifikasi memakai koordinat gambar; perspektif CCTV dapat mengubah sudut dan perpindahan lateral."]),
            "rule_reason": reason,
        })
        return result
