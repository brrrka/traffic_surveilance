"""Bilingual research matrix; no universal pixels-to-meters constant."""
import math


def _translate_environment(meta):
    maps = [
        ('weather', {'Unknown':'Tidak diketahui','Clear':'Cerah','Overcast':'Berawan','Rain':'Hujan','Fog':'Kabut'}),
        ('lighting', {'Unknown':'Tidak diketahui','Daylight':'Siang hari','Dusk/Dawn':'Senja/Fajar','Night':'Malam hari'}),
        ('road_surface', {'Unknown':'Tidak diketahui','Dry':'Kering','Wet':'Basah','Snow/Ice':'Salju/Es'}),
    ]
    labels=[]
    for key,table in maps:
        value=str(meta.get(key,'Unknown'))
        labels.append(table.get(value,value))
    return f'Cuaca: {labels[0]}; Cahaya: {labels[1]}; Jalan: {labels[2]}'


# Dua keluaran berbeda: average selama track pada video dan pre-impact 0,8 s.
# Semua sampel yang diringkas harus punya satuan/status pengukuran yang sah.
PRE_IMPACT_WINDOW_S = 0.8
VALID_METRIC_STATUSES = ('calibrated_reference', 'estimated_prior')


def _valid_rows(vehicle, samples=None):
    return list(samples if samples is not None else vehicle.get('speed_history', []))


def _mean_of_speeds(rows, key, min_samples=2):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return round(sum(values) / len(values), 2) if len(values) >= min_samples else None


def _average_video_speed(vehicle, samples=None):
    """Arithmetic mean across all observed VALID km/h frames in this track."""
    rows = [row for row in _valid_rows(vehicle, samples)
            if row.get('speed_status') in VALID_METRIC_STATUSES and row.get('speed_kmh') is not None]
    modes = {row.get('speed_status') for row in rows}
    # Never mix calibrated metres and assumed vehicle-length metres.
    if len(modes) != 1:
        return None, 0
    value = _mean_of_speeds(rows, 'speed_kmh')
    count = 0
    for row in rows:
        try:
            numeric = float(row['speed_kmh'])
            count += int(math.isfinite(numeric) and numeric >= 0)
        except (TypeError, ValueError, KeyError):
            pass
    return value, count if value is not None else 0


def _pre_event_rows(vehicle, event_frame, fps, window=PRE_IMPACT_WINDOW_S, samples=None):
    start = int(event_frame) - max(1, int(float(fps) * window))
    return [row for row in _valid_rows(vehicle, samples)
            if start <= int(row.get('frame', -1)) < int(event_frame)]


def _speed_in_window(vehicle, event_frame, fps, key, window=PRE_IMPACT_WINDOW_S, samples=None):
    """Arithmetic mean of valid pre-contact samples (not the whole-video mean)."""
    rows = _pre_event_rows(vehicle, event_frame, fps, window, samples)
    if key == 'speed_kmh':
        rows = [row for row in rows if row.get('speed_status') in VALID_METRIC_STATUSES]
        if len({row.get('speed_status') for row in rows if row.get(key) is not None}) != 1:
            return None
    return _mean_of_speeds(rows, key)


def _status_in_window(vehicle, event_frame, fps, window=PRE_IMPACT_WINDOW_S, samples=None):
    statuses = set()
    for row in _pre_event_rows(vehicle, event_frame, fps, window, samples):
        vec = row.get('world_velocity_m_s') or row.get('velocity_m_s')
        if row.get('speed_kmh') is None or vec is None or len(vec) < 2:
            continue
        if row.get('speed_status') in VALID_METRIC_STATUSES:
            statuses.add(row['speed_status'])
    if len(statuses) == 1:
        return next(iter(statuses))
    return 'mixed' if statuses else 'uncalibrated'


def _metric_velocity_in_window(vehicle, event_frame, fps, window=PRE_IMPACT_WINDOW_S,
                               samples=None, required_status=None, shared_frames=None):
    """Mean vector of synchronized pre-contact metric-speed samples."""
    vectors = []
    for row in _pre_event_rows(vehicle, event_frame, fps, window, samples):
        if required_status and row.get('speed_status') != required_status:
            continue
        if shared_frames is not None and int(row.get('frame', -1)) not in shared_frames:
            continue
        vec = row.get('world_velocity_m_s') or row.get('velocity_m_s')
        if vec is None or len(vec) < 2:
            continue
        try:
            vx, vy = float(vec[0]), float(vec[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(vx) and math.isfinite(vy):
            vectors.append((vx, vy))
    if len(vectors) < 2:
        return None
    return (sum(v[0] for v in vectors) / len(vectors),
            sum(v[1] for v in vectors) / len(vectors))


def _shared_metric_frames(v1, v2, event_frame, fps, samples1=None, samples2=None, status=None):
    def frames(vehicle, samples):
        return {int(row['frame']) for row in _pre_event_rows(vehicle, event_frame, fps, samples=samples)
                if row.get('speed_status') == status and (row.get('world_velocity_m_s') or row.get('velocity_m_s')) is not None}
    return frames(v1, samples1) & frames(v2, samples2)


def _footpoint_at_or_before(vehicle, event_frame):
    frames = vehicle.get('frame_indices', [])
    points = vehicle.get('footpoint_trajectory', [])
    candidates = [(int(f), p) for f, p in zip(frames, points) if int(f) <= int(event_frame) and p is not None]
    return max(candidates, key=lambda row: row[0])[1] if candidates else None



def _closing_speed_kmh(v1, v2, p1, p2):
    """Closing component along the line of centers, in km/h."""
    if v1 is None or v2 is None or p1 is None or p2 is None:
        return None
    try:
        rx = float(p2[0]) - float(p1[0])
        ry = float(p2[1]) - float(p1[1])
        distance = math.hypot(rx, ry)
        if distance <= 1e-6:
            return None
        relative_vx = float(v2[0]) - float(v1[0])
        relative_vy = float(v2[1]) - float(v1[1])
        closing_m_s = -((relative_vx * rx + relative_vy * ry) / distance)
        if not math.isfinite(closing_m_s):
            return None
        return round(max(0.0, closing_m_s) * 3.6, 2)
    except (TypeError, ValueError, IndexError):
        return None


def _kinematic_risk_from_closing_speed(closing_speed_kmh):
    """Indicative software-test band; not injury/damage severity."""
    if closing_speed_kmh is None:
        return None
    value = float(closing_speed_kmh)
    if not math.isfinite(value) or value < 0:
        return None
    if value < 20.0:
        return "Rendah"
    if value < 50.0:
        return "Sedang"
    return "Tinggi"


def build_empty_matrix_13(metadata=None, calibration=None):
    m=metadata or {}
    return {
        'crash_type_code':None,
        'crash_type':'Belum terdeteksi kandidat kontak / No candidate contact',
        'road_user_pair':'Car – Car',
        'vehicle1_id':None,'vehicle2_id':None,
        'vehicle1_role':'Belum ditentukan','vehicle2_role':'Belum ditentukan',
        'vehicle1_maneuver':'Belum tersedia / Unavailable','vehicle2_maneuver':'Belum tersedia / Unavailable',
        'vehicle1_pre_crash_speed_kmh':None,'vehicle2_pre_crash_speed_kmh':None,
        'vehicle1_display_speed_kmh':None,'vehicle2_display_speed_kmh':None,
        'vehicle1_display_speed_source':'unavailable','vehicle2_display_speed_source':'unavailable',
        'vehicle1_display_speed_source_label':'tidak tersedia','vehicle2_display_speed_source_label':'tidak tersedia',
        'vehicle1_display_speed_sample_count':0,'vehicle2_display_speed_sample_count':0,
        'vehicle1_speed_diagnostics':{}, 'vehicle2_speed_diagnostics':{},
        'vehicle1_average_speed_kmh':None,'vehicle2_average_speed_kmh':None,
        'vehicle1_speed_source':'unavailable','vehicle2_speed_source':'unavailable',
        'vehicle1_speed_source_label':'tidak tersedia','vehicle2_speed_source_label':'tidak tersedia',
        'vehicle1_speed_sample_count':0,'vehicle2_speed_sample_count':0,
        'pair_speed_source':'unavailable','pair_speed_source_label':'tidak tersedia',
        'pair_speed_sample_count':0,'pair_speed_status':'unavailable',
        'vehicle1_average_sample_count':0,'vehicle2_average_sample_count':0,
        'speed_aggregation_method':'arithmetic_mean',
        'pre_impact_window_seconds':PRE_IMPACT_WINDOW_S,
        'vehicle1_pre_crash_speed_px_s':None,'vehicle2_pre_crash_speed_px_s':None,
        'relative_speed_kmh':None,'relative_speed_px_s':None,
        'impact_speed_kmh':None,'impact_speed_px_s':None,'impact_speed_status':'uncalibrated',
        'impact_speed_method':'not_available','impact_speed_confidence_label':'Tidak tersedia','impact_angle_deg':None,
        'evasive_action':'Tidak dinilai: belum ditemukan pasangan kandidat kontak',
        'point_of_impact':'Tidak diketahui / Unknown',
        'crash_severity':'Belum tersedia — belum ada kandidat kontak',
        'severity_category':None,
        'severity_basis':'Severity indikatif memerlukan Impact Speed dan kandidat kontak; estimasi prior harus diberi label keyakinan rendah.',
        'weather_conditions':_translate_environment(m),
        'classification_confidence':'Tidak tersedia','classification_evidence':[],
        'role_assignment_reason':'Tidak ada pasangan yang teridentifikasi.',
        'speed_status':'uncalibrated',
        'speed_calibration_warning':'Tanpa anchor metrik yang valid, hanya kecepatan citra (px/s) yang tersedia.',
        'relative_speed_definition':'Besaran selisih vektor kecepatan pada bidang jalan (bila tersedia).',
        'impact_speed_definition':'Komponen closing speed pra-kontak pada garis penghubung kedua kendaraan; bukan delta-v atau kecepatan benturan fisik.',
        'impact_angle_reference':'image_plane',
        'classification_coordinate_space':'image_px',
        'impact_angle_definition':'Sudut antara arah lintasan pada bidang gambar, bukan sudut benturan fisik 3D.',
    }



# Conservative cascading fallback. Only samples STRICTLY before the candidate
# event are eligible. Never silently substitute post-contact video averages.
FALLBACK_WINDOWS_S = (PRE_IMPACT_WINDOW_S, 1.0, 2.0)


def _precontact_rows(vehicle, event_frame, samples=None, window=None, fps=None):
    rows = _valid_rows(vehicle, samples)
    lower = (int(event_frame) - max(1, int(float(fps) * window))) if window is not None else None
    return [row for row in rows if isinstance(row, dict)
            and row.get('frame') is not None
            and (lower is None or lower <= int(row['frame']))
            and int(row['frame']) < int(event_frame)]


def _source_label(window):
    if window == PRE_IMPACT_WINDOW_S:
        return 'pre_impact_mean_0_8s'
    if window is None:
        return 'fallback_mean_all_precontact'
    return f'fallback_mean_{window:.1f}s'.replace('.', '_')


def _window_label(window):
    return 'seluruh riwayat sebelum kontak' if window is None else f'{window:.1f} detik sebelum kontak'


def _select_scalar_speed(vehicle, event_frame, fps, samples=None):
    """Pick first valid window; prefer calibrated over prior within each window."""
    for window in (*FALLBACK_WINDOWS_S, None):
        rows = _precontact_rows(vehicle, event_frame, samples, window, fps)
        for status in VALID_METRIC_STATUSES:
            eligible = [row for row in rows if row.get('speed_status') == status]
            value = _mean_of_speeds(eligible, 'speed_kmh', min_samples=2)
            if value is not None:
                count = 0
                for row in eligible:
                    try:
                        x = float(row.get('speed_kmh'))
                        count += int(math.isfinite(x) and x >= 0)
                    except (TypeError, ValueError):
                        continue
                return {'value': value, 'source': _source_label(window),
                        'source_label': _window_label(window), 'status': status,
                        'sample_count': count}
    return {'value': None, 'source': 'unavailable', 'source_label': 'tidak tersedia',
            'status': 'uncalibrated', 'sample_count': 0}


def _valid_vector(row):
    vec = row.get('world_velocity_m_s') or row.get('velocity_m_s')
    if vec is None or len(vec) < 2:
        return None
    try:
        vx, vy = float(vec[0]), float(vec[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (vx, vy) if math.isfinite(vx) and math.isfinite(vy) else None


# At most one frame at the video's native FPS; never pair after-contact samples.
# A one-frame offset is an *approximation*, not proof that both observations
# refer to exactly the same instant.
PAIR_SYNC_TOLERANCE_FRAMES = 5


def _valid_vector_rows(rows, status):
    out = {}
    for row in rows:
        if row.get('speed_status') != status:
            continue
        vec = _valid_vector(row)
        if vec is None:
            continue
        try:
            f = int(row['frame'])
        except (KeyError, TypeError, ValueError):
            continue
        out[f] = (row, vec)
    return out


def _nearest_pairs(lookup1, lookup2, tolerance_frames):
    """Greedy one-to-one nearest frame matching; no fabricated observations."""
    pairs = []
    remaining = set(lookup2)
    for f1 in sorted(lookup1):
        viable = [f2 for f2 in remaining if abs(f2 - f1) <= tolerance_frames]
        if not viable:
            continue
        f2 = min(viable, key=lambda value: (abs(value-f1), value))
        pairs.append((f1, f2))
        remaining.remove(f2)
    return pairs


def _matched_vectors(v1, v2, frame, fps, samples1=None, samples2=None):
    """Compare pre-contact velocity vectors on the same / adjacent frames.

    Exact-frame pairing is preferred. A bounded +/-1-frame pairing is used
    only when >=2 exact-frame pairs cannot be found. Scale provenance must
    match; the vehicle-length prior is explicitly low confidence.
    """
    for window in (*FALLBACK_WINDOWS_S, None):
        rows1 = _precontact_rows(v1, frame, samples1, window, fps)
        rows2 = _precontact_rows(v2, frame, samples2, window, fps)
        for status in VALID_METRIC_STATUSES:
            lookup1 = _valid_vector_rows(rows1, status)
            lookup2 = _valid_vector_rows(rows2, status)
            common = sorted(set(lookup1) & set(lookup2))
            if len(common) >= 2:
                frame_pairs = [(f, f) for f in common]
            else:
                frame_pairs = _nearest_pairs(lookup1, lookup2, PAIR_SYNC_TOLERANCE_FRAMES)
            if len(frame_pairs) < 2:
                continue
            va = [lookup1[f1][1] for f1, f2 in frame_pairs]
            vb = [lookup2[f2][1] for f1, f2 in frame_pairs]
            avg1 = (sum(v[0] for v in va) / len(va), sum(v[1] for v in va) / len(va))
            avg2 = (sum(v[0] for v in vb) / len(vb), sum(v[1] for v in vb) / len(vb))
            f1, f2 = frame_pairs[-1]
            sync_delta = max(abs(a-b) for a,b in frame_pairs)
            return {
                'v1': avg1, 'v2': avg2, 'frame': max(f1, f2),
                'frame1': f1, 'frame2': f2,
                'status': status, 'source': _source_label(window),
                'source_label': _window_label(window), 'sample_count': len(frame_pairs),
                'row1': lookup1[f1][0], 'row2': lookup2[f2][0],
                'sync_method': 'exact' if sync_delta == 0 else 'nearest_frame_1',
                'sync_max_gap_frames': sync_delta,
                'sync_max_gap_s': round(sync_delta / float(fps), 4),
            }
    return None


def _min_frame_distance(sorted_a, sorted_b):
    """O(n+m) nearest difference, safe for long CCTV videos."""
    i = j = 0
    gap = None
    while i < len(sorted_a) and j < len(sorted_b):
        diff = abs(sorted_a[i] - sorted_b[j])
        gap = diff if gap is None else min(gap, diff)
        if gap == 0:
            break
        if sorted_a[i] <= sorted_b[j]:
            i += 1
        else:
            j += 1
    return gap


def _pair_vector_diagnostic(v1, v2, frame, samples1=None, samples2=None):
    """Explain why V1/V2 scalar values may exist without pair velocity."""
    rows1 = _precontact_rows(v1, frame, samples1)
    rows2 = _precontact_rows(v2, frame, samples2)
    data = {}
    for status in VALID_METRIC_STATUSES:
        a = _valid_vector_rows(rows1, status)
        b = _valid_vector_rows(rows2, status)
        exact_count = len(set(a) & set(b))
        near_count = len(_nearest_pairs(a, b, PAIR_SYNC_TOLERANCE_FRAMES))
        gap = _min_frame_distance(sorted(a), sorted(b))
        data[status] = {
            'v1_vector_frames': len(a), 'v2_vector_frames': len(b),
            'v1_first_frame': min(a) if a else None,
            'v1_last_frame': max(a) if a else None,
            'v2_first_frame': min(b) if b else None,
            'v2_last_frame': max(b) if b else None,
            'exact_pairs': exact_count,
            'near_pairs_1_frame': near_count,
            'min_frame_gap': gap,
        }
    if all(x['v1_vector_frames'] == 0 for x in data.values()):
        reason = 'V1 tidak memiliki vektor kecepatan metrik valid sebelum kontak.'
    elif all(x['v2_vector_frames'] == 0 for x in data.values()):
        reason = 'V2 tidak memiliki vektor kecepatan metrik valid sebelum kontak.'
    elif all(x['v1_vector_frames'] == 0 or x['v2_vector_frames'] == 0 for x in data.values()):
        reason = 'Vektor V1 dan V2 hanya tersedia dengan status/skala yang berbeda.'
    elif any(x['near_pairs_1_frame'] == 1 for x in data.values()) and not any(x['near_pairs_1_frame'] >= 2 for x in data.values()):
        reason = 'Hanya satu pasangan vektor tersedia; diperlukan minimal dua.'
    else:
        reason = 'Tidak ada dua pasangan vektor pra-kontak yang berdekatan dalam batas 1 frame.'
    return {'reason':reason, 'by_status':data,
            'max_sync_tolerance_frames':PAIR_SYNC_TOLERANCE_FRAMES}


def _aligned_impact_positions(matched, fp1, fp2, calibration, fps):
    """Align positions to the later matched time under short constant velocity.

    For prior-estimated speeds this is only an image-space approximation.
    Missing correction data means impact remains unavailable (no guessing).
    """
    if fp1 is None or fp2 is None:
        return None, None, 'footpoint_missing'
    status = matched['status']
    if status == 'calibrated_reference':
        if calibration is None:
            return None, None, 'camera_calibration_missing'
        p1 = calibration.ground_point(fp1)
        p2 = calibration.ground_point(fp2)
    else:
        p1, p2 = fp1, fp2
    if p1 is None or p2 is None:
        return None, None, 'projection_missing'
    if matched['frame1'] == matched['frame2']:
        return p1, p2, 'exact_frame'
    later = max(matched['frame1'], matched['frame2'])
    adjusted = []
    for idx, point, at in ((1,p1,matched['frame1']),(2,p2,matched['frame2'])):
        row = matched['row1'] if idx == 1 else matched['row2']
        v = _valid_vector(row)
        if v is None:
            return None, None, 'velocity_missing_for_alignment'
        dt = (later - at) / float(fps)
        if status == 'estimated_prior':
            try:
                scale = float(row.get('estimated_scale_m_per_px'))
            except (TypeError, ValueError):
                return None, None, 'prior_scale_missing_for_alignment'
            if not math.isfinite(scale) or scale <= 0:
                return None, None, 'prior_scale_invalid_for_alignment'
            # Convert assumed metric velocity back to image px/s before
            # projecting pixel footpoints to the same instant.
            v = (v[0] / scale, v[1] / scale)
        adjusted.append((float(point[0])+v[0]*dt, float(point[1])+v[1]*dt))
    return adjusted[0], adjusted[1], 'constant_velocity_1_frame'


def _paired_footpoint(vehicle, frame, sample_row):
    """Use same-frame position as matched velocity, including full-engine samples."""
    point = sample_row.get('footpoint_px')
    if point is not None and len(point) >= 2:
        return point
    for f, p in zip(vehicle.get('frame_indices', []), vehicle.get('footpoint_trajectory', [])):
        if int(f) == frame and p is not None:
            return p
    return None




def _display_speed_with_labeled_video_fallback(precontact_choice, whole_video_average, whole_video_count):
    """UI fallback only: never treat a post-contact average as pre-impact evidence."""
    if precontact_choice['value'] is not None:
        return dict(precontact_choice)
    if whole_video_average is not None and whole_video_count >= 2:
        return {
            'value': whole_video_average,
            'source': 'fallback_mean_whole_video_not_preimpact',
            'source_label': 'rata-rata seluruh video (bukan kecepatan pra-kontak)',
            'status': 'illustrative_only', 'sample_count': whole_video_count,
        }
    return dict(precontact_choice)


def _speed_availability_diagnostic(vehicle, event_frame, samples=None):
    rows = _valid_rows(vehicle, samples)
    valid = []
    for row in rows:
        if row.get('speed_status') not in VALID_METRIC_STATUSES:
            continue
        try:
            v = float(row.get('speed_kmh'))
            if math.isfinite(v) and v >= 0:
                valid.append(int(row['frame']))
        except (TypeError, ValueError, KeyError):
            continue
    before = [f for f in valid if f < event_frame]
    after = [f for f in valid if f >= event_frame]
    return {
        'event_frame': int(event_frame),
        'valid_precontact_samples': len(before),
        'valid_at_or_after_event_samples': len(after),
        'first_metric_frame': min(valid) if valid else None,
        'last_metric_frame': max(valid) if valid else None,
        'reason': (
            'Tidak ada sampel km/jam valid sebelum kandidat kontak. Cek event_frame, ID tracking, dan kalibrasi.'
            if len(before) == 0 else
            'Hanya satu sampel km/jam valid sebelum kontak; minimum dua sampel diperlukan.'
            if len(before) == 1 else
            'Ada sampel pra-kontak; status kalibrasi atau validitas sampel perlu diperiksa.'
        ),
    }

def build_matrix_13(event, histories, fps, metadata=None, calibration=None, vehicle_speed_samples=None):
    m=metadata or {}
    v1=histories[int(event['v1'])];v2=histories[int(event['v2'])]
    frame=int(event['event_frame'])
    # Full-history samples are supplied by VehicleSpeedEngine, because
    # TrajectoryManager keeps only the latest max_points observations.
    samples1 = (vehicle_speed_samples or {}).get(int(event['v1'])) if vehicle_speed_samples is not None else None
    samples2 = (vehicle_speed_samples or {}).get(int(event['v2'])) if vehicle_speed_samples is not None else None
    average1, average_count1 = _average_video_speed(v1, samples1)
    average2, average_count2 = _average_video_speed(v2, samples2)
    selection1 = _select_scalar_speed(v1, frame, fps, samples=samples1)
    selection2 = _select_scalar_speed(v2, frame, fps, samples=samples2)
    # Keep raw precontact fields untouched for scientific traceability.
    display1 = _display_speed_with_labeled_video_fallback(selection1, average1, average_count1)
    display2 = _display_speed_with_labeled_video_fallback(selection2, average2, average_count2)
    diag1 = _speed_availability_diagnostic(v1, frame, samples1)
    diag2 = _speed_availability_diagnostic(v2, frame, samples2)
    speed1, speed2 = selection1['value'], selection2['value']
    # Pixel-speed remains diagnostic only: it is never relabelled km/h.
    speed1_px = _speed_in_window(v1, frame, fps, 'speed_px_s', samples=samples1)
    speed2_px = _speed_in_window(v2, frame, fps, 'speed_px_s', samples=samples2)
    status1, status2 = selection1['status'], selection2['status']
    if status1 == status2 and status1 in VALID_METRIC_STATUSES:
        status = status1
    elif status1 in VALID_METRIC_STATUSES or status2 in VALID_METRIC_STATUSES:
        status = 'mixed'
    else:
        status = 'uncalibrated'

    relative = None
    closing = None
    relative_px = round(float(event.get('relative_speed_px_s')), 2) if event.get('relative_speed_px_s') is not None else None
    closing_px = round(float(event.get('closing_speed_px_s')), 2) if event.get('closing_speed_px_s') is not None else None
    impact_method = 'not_available'
    impact_confidence_label = 'Tidak tersedia'
    physics_source = 'unavailable'
    physics_source_label = 'tidak tersedia'
    physics_sample_count = 0
    # Pair physics uses SAME frames and SAME measurement basis for both cars.
    # Individual scalar fallbacks do not supply the vectors needed for physics.
    matched = _matched_vectors(v1, v2, frame, fps, samples1, samples2)
    pair_vector_diag = _pair_vector_diagnostic(v1, v2, frame, samples1, samples2)
    alignment_status = 'unavailable'
    if matched is not None:
        v_a, v_b = matched['v1'], matched['v2']
        physics_source, physics_source_label = matched['source'], matched['source_label']
        physics_sample_count = matched['sample_count']
        relative = round(math.hypot(v_a[0] - v_b[0], v_a[1] - v_b[1]) * 3.6, 2)
        fp1 = _paired_footpoint(v1, matched['frame1'], matched['row1'])
        fp2 = _paired_footpoint(v2, matched['frame2'], matched['row2'])
        p1, p2, alignment_status = _aligned_impact_positions(
            matched, fp1, fp2, calibration, fps
        )
        if matched['status'] == 'calibrated_reference':
            closing = _closing_speed_kmh(v_a, v_b, p1, p2)
            if closing is not None:
                impact_method = 'ground_plane_homography_closing_component'
                impact_confidence_label = (
                    'Sedang' if physics_source == 'pre_impact_mean_0_8s'
                    and matched['sync_method'] == 'exact' else 'Rendah (fallback/sinkronisasi)'
                )
        elif matched['status'] == 'estimated_prior':
            closing = _closing_speed_kmh(v_a, v_b, p1, p2)
            if closing is not None:
                impact_method = 'low_confidence_vehicle_dimension_prior'
                impact_confidence_label = 'Rendah'

    ev=[]
    for label,v in (('V1',v1),('V2',v2)):
        samples=[row for row in v.get('speed_history',[]) if frame-int(fps*1.0)<=row['frame']<frame]
        if len(samples)>=7:
            first=samples[0].get('speed_px_s');last=samples[-1].get('speed_px_s')
            if first is not None and last is not None and first>20 and last<first*0.60:
                ev.append(f'{label}: indikasi perlambatan citra (bukan bukti pengereman)')
    sides=f"V1: {event.get('contact_side_v1','Tidak diketahui')}; V2: {event.get('contact_side_v2','Tidak diketahui')}"
    location=event.get('location')
    if location:
        poi=f"Perkiraan area tumpang tindih citra x={location[0]}, y={location[1]}; {sides}"
    else:
        poi='Tidak diketahui / Unknown'
    manual_severity = m.get('severity_reference', 'Unknown')
    if manual_severity and manual_severity not in ('Unknown', 'Tidak diketahui'):
        severity = f'{manual_severity} (berdasarkan metadata manual)'
        severity_category = str(manual_severity)
        severity_basis = 'Menggunakan severity_reference yang dimasukkan secara manual; bukan hasil inferensi CCTV.'
    else:
        severity_category = _kinematic_risk_from_closing_speed(closing)
        if severity_category is None:
            severity = 'Belum tersedia — Impact Speed belum dapat dihitung'
            severity_basis = (
                'Severity indikatif memerlukan dua vektor kecepatan pra-kontak dan posisi kedua kendaraan '
                'yang cukup untuk menghitung closing speed.'
            )
        elif matched is not None and (matched['status'] == 'estimated_prior' or physics_source != 'pre_impact_mean_0_8s' or matched['sync_method'] != 'exact'):
            severity = f'{severity_category} (indikasi risiko kinematik — fallback, keyakinan rendah)'
            severity_basis = (
                f'Berdasarkan estimasi closing speed {closing:.2f} km/jam menggunakan rerata {physics_source_label}. '
                'Ini bukan pengukuran sesaat saat tumbukan atau tingkat cedera/kerusakan aktual.'
            )
        else:
            severity = f'{severity_category} (indikasi risiko kinematik)'
            severity_basis = (
                f'Berdasarkan closing speed pra-kontak {closing:.2f} km/jam. '
                'Kategori ini hanya band indikatif, bukan tingkat cedera atau kerusakan kendaraan.'
            )
    if matched is not None and matched['status'] == 'calibrated_reference':
        warning = 'Referensi meter tersedia; masih terdapat ketidakpastian tracking dan lokasi footpoint.'
    elif matched is not None and matched['status'] == 'estimated_prior':
        warning = (
            'ESTIMASI KEYAKINAN RENDAH: skala memakai panjang kendaraan generik 4,4 m. '
            'Impact Speed dihitung sebagai komponen closing dari vektor estimasi pada bidang citra; '
            'bukan hasil kalibrasi ground-plane.'
        )
    elif status == 'mixed':
        warning = 'Basis kecepatan V1/V2 berbeda atau data vektor tidak sinkron; gunakan sumber masing-masing dan cek data pasangan.'
    else:
        warning = 'Dua vektor metrik serempak belum cukup; Relative/Impact Speed mungkin tidak tersedia walau salah satu kecepatan kendaraan ada.'
    if matched is not None and matched['sync_method'] != 'exact':
        warning += (' Vektor V1/V2 dicocokkan dengan toleransi 1 frame; posisi '
                    'diperkirakan ke waktu yang sama dengan asumsi kecepatan konstan. '
                    'Ini bukan pengukuran simultan langsung.')
    if physics_source not in ('unavailable', 'pre_impact_mean_0_8s'):
        warning += f' Perhitungan pasangan memakai rerata {physics_source_label}, bukan nilai sesaat sebelum kontak.'
    return {
        'crash_type_code':event.get('crash_type_code'),
        'crash_type':event.get('crash_type','Belum dapat diklasifikasikan / Unclassified'),
        'road_user_pair':'Car – Car',
        'vehicle1_id':int(event['v1']),'vehicle2_id':int(event['v2']),
        'vehicle1_role':event.get('vehicle1_role','Mobil 1'),
        'vehicle2_role':event.get('vehicle2_role','Mobil 2'),
        'vehicle1_maneuver':event.get('maneuver_v1','Lintasan belum cukup'),
        'vehicle2_maneuver':event.get('maneuver_v2','Lintasan belum cukup'),
        'vehicle1_pre_crash_speed_kmh':speed1,
        'vehicle2_pre_crash_speed_kmh':speed2,
        'vehicle1_display_speed_kmh':display1['value'],
        'vehicle2_display_speed_kmh':display2['value'],
        'vehicle1_display_speed_source':display1['source'],
        'vehicle2_display_speed_source':display2['source'],
        'vehicle1_display_speed_source_label':display1['source_label'],
        'vehicle2_display_speed_source_label':display2['source_label'],
        'vehicle1_display_speed_sample_count':display1['sample_count'],
        'vehicle2_display_speed_sample_count':display2['sample_count'],
        'vehicle1_speed_diagnostics':diag1 if speed1 is None else {},
        'vehicle2_speed_diagnostics':diag2 if speed2 is None else {},
        'vehicle1_speed_source':selection1['source'],
        'vehicle2_speed_source':selection2['source'],
        'vehicle1_speed_source_label':selection1['source_label'],
        'vehicle2_speed_source_label':selection2['source_label'],
        'vehicle1_speed_sample_count':selection1['sample_count'],
        'vehicle2_speed_sample_count':selection2['sample_count'],
        'pair_speed_source':physics_source,
        'pair_speed_source_label':physics_source_label,
        'pair_speed_sample_count':physics_sample_count,
        'pair_speed_status':matched['status'] if matched is not None else 'unavailable',
        'pair_speed_sync_method': matched['sync_method'] if matched else 'unavailable',
        'pair_speed_sync_max_gap_frames': matched['sync_max_gap_frames'] if matched else None,
        'pair_speed_sync_max_gap_seconds': matched['sync_max_gap_s'] if matched else None,
        'pair_speed_position_alignment':alignment_status,
        'pair_speed_vector_diagnostics':pair_vector_diag,
        'pair_speed_diagnostic': (
            pair_vector_diag['reason'] + ' Rata-rata kecepatan skalar tidak cukup untuk menghitung fisika pasangan.'
            if matched is None else
            ('Pasangan vektor ditemukan dengan toleransi satu frame; estimasi berketidakpastian lebih tinggi.'
             if matched['sync_method'] != 'exact' else '')
        ),
        'vehicle1_average_speed_kmh':average1,
        'vehicle2_average_speed_kmh':average2,
        'vehicle1_average_sample_count':average_count1,
        'vehicle2_average_sample_count':average_count2,
        'speed_aggregation_method':'arithmetic_mean',
        'pre_impact_window_seconds':PRE_IMPACT_WINDOW_S,
        'vehicle1_pre_crash_speed_px_s':speed1_px,
        'vehicle2_pre_crash_speed_px_s':speed2_px,
        'relative_speed_kmh':relative,
        'relative_speed_px_s':relative_px,
        'impact_speed_kmh':closing,
        'impact_speed_px_s':closing_px,
        'impact_speed_status':status,
        'impact_speed_method':impact_method,
        'impact_speed_confidence_label':impact_confidence_label,
        'impact_angle_deg':event.get('impact_angle_deg'),
        'evasive_action':'; '.join(ev) if ev else 'Belum ada bukti kuat manuver menghindar',
        'point_of_impact':poi,
        'crash_severity':severity,
        'severity_category':severity_category,
        'severity_basis':severity_basis,
        'weather_conditions':_translate_environment(m),
        'classification_confidence':event.get('classification_confidence','Rendah'),
        'classification_evidence':event.get('classification_evidence',[]),
        'role_assignment_reason':event.get('role_assignment_reason',''),
        'speed_status':status,
        'vehicle1_speed_status':status1,
        'vehicle2_speed_status':status2,
        'speed_calibration_warning':warning,
        'impact_speed_warning':warning,
        'relative_speed_definition':'Norma selisih vektor kecepatan pada bidang jalan.',
        'impact_speed_definition':'Komponen closing speed pra-kontak pada garis penghubung kedua kendaraan; bukan delta-v atau kecepatan benturan fisik.',
        'impact_angle_reference':event.get('impact_angle_reference','image_plane'),
        'classification_coordinate_space':event.get('classification_coordinate_space','image_px'),
        'impact_angle_definition':('Sudut antar arah lintasan pada bidang jalan hasil homography; bukan sudut benturan fisik 3D.' if event.get('impact_angle_reference') == 'ground_plane' else 'Sudut antar arah lintasan pada bidang gambar; bukan sudut benturan fisik 3D.'),
    }


def build_speed_profile(event, histories, fps, vehicle_speed_samples=None):
    a=histories[int(event['v1'])];b=histories[int(event['v2'])]
    # Use full Speed Engine samples to avoid losing an early contact event
    # when TrajectoryManager's bounded history later discards old frames.
    h1={row['frame']:row for row in (vehicle_speed_samples or {}).get(int(event['v1']), a.get('speed_history',[]))}
    h2={row['frame']:row for row in (vehicle_speed_samples or {}).get(int(event['v2']), b.get('speed_history',[]))}
    at=int(event['event_frame']);start=at-int(fps*2.0)
    frames=[f for f in sorted(set(h1)&set(h2)) if start<=f<=at]
    frames=frames[::max(1,len(frames)//35)]
    result=[]
    for f in frames:
        result.append({'time':f'{(f-at)/fps:.2f}s',
                       'frame':f,
                       'v1Speed':h1[f].get('speed_kmh'),
                       'v2Speed':h2[f].get('speed_kmh'),
                       'v1PixelSpeed':h1[f].get('speed_px_s'),
                       'v2PixelSpeed':h2[f].get('speed_px_s'),
                       'v1SpeedStatus':h1[f].get('speed_status','uncalibrated'),
                       'v2SpeedStatus':h2[f].get('speed_status','uncalibrated')})
    return result
