from collections import deque
from typing import Deque, Dict, List, Tuple

from .utils import box_diag, center_of_box, distance


def compute_risk_events(
    tracks: Dict[int, dict],
    genders: Dict[int, str],
    centers_hist: Dict[int, Deque[Tuple[int, int]]],
    frame_shape,
    approach_check_frames: int = 6,
) -> Tuple[List[str], int]:
    """
    Heuristics to flag potentially risky situations:
      - Lone woman surrounded by multiple men nearby
      - Rapid approach of a man towards a woman
      - Possibly fallen person (very low h/w aspect ratio)
    """
    H, W = frame_shape[:2]
    diag_frame = (W**2 + H**2) ** 0.5

    female_ids = [tid for tid, g in genders.items() if g == "F" and tid in tracks]
    male_ids = [tid for tid, g in genders.items() if g == "M" and tid in tracks]

    events: List[str] = []
    risk_score = 0

    centers = {tid: center_of_box(tr["xyxy"]) for tid, tr in tracks.items()}
    diags = {tid: box_diag(tr["xyxy"]) for tid, tr in tracks.items()}

    # 1) Lone woman surrounded by multiple men in close proximity
    for fid in female_ids:
        f_center = centers[fid]
        f_diag = diags[fid]
        prox_thresh = max(0.06 * diag_frame, 0.75 * f_diag)

        nearby_males = []
        nearby_females = []
        for mid in male_ids:
            if distance(f_center, centers[mid]) < prox_thresh:
                nearby_males.append(mid)
        for fid2 in female_ids:
            if fid2 == fid:
                continue
            if distance(f_center, centers[fid2]) < prox_thresh:
                nearby_females.append(fid2)

        if len(nearby_males) >= 2 and len(nearby_females) == 0:
            events.append(f"Female {fid} surrounded by {len(nearby_males)} males in close proximity")
            risk_score += 3

    # 2) Rapid approach: man->woman distance decreases fast and is currently close
    for mid in male_ids:
        for fid in female_ids:
            if len(centers_hist.get(mid, [])) < approach_check_frames or len(centers_hist.get(fid, [])) < approach_check_frames:
                continue
            d_now = distance(centers[mid], centers[fid])
            d_past = distance(centers_hist[mid][0], centers_hist[fid][0])

            decrease = d_past - d_now
            close_thresh = 0.05 * diag_frame
            approach_thresh = 0.04 * diag_frame

            if d_now < close_thresh and decrease > approach_thresh:
                events.append(f"Male {mid} rapidly approaching Female {fid}")
                risk_score += 2

    # 3) Possibly fallen person: height/width ratio very small on a large box
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = tr["xyxy"]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        aspect = h / w
        area = w * h
        if aspect < 0.55 and area > 0.02 * (W * H):
            events.append(f"Track {tid} possibly lying/fallen")
            risk_score += 3

    return events, risk_score


def risk_level(score: int) -> str:
    if score >= 5:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"