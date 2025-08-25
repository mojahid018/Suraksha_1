from typing import Dict, List, Tuple

from insightface.app import FaceAnalysis

from .utils import center_of_box, distance, point_in_box


def _map_gender(face) -> Tuple[str, float]:
    """
    Map InsightFace face.gender to ('M' or 'F') and a confidence-like score.
    InsightFace: gender=0 (female), 1 (male). We use det_score as a proxy confidence.
    """
    g = getattr(face, "gender", None)
    det_score = float(getattr(face, "det_score", 0.5))
    if g is None:
        s = getattr(face, "sex", None)
        if isinstance(s, str):
            s = s.upper()
            if s.startswith("M"):
                return "M", det_score
            if s.startswith("F"):
                return "F", det_score
        return "U", det_score
    try:
        g = float(g)
    except Exception:
        return "U", det_score
    return ("M" if g >= 0.5 else "F"), det_score


class GenderEstimator:
    def __init__(self, providers=None, name: str = "buffalo_l", det_size=(640, 640)):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)

    def get_faces(self, frame) -> List:
        """
        Run face detection/attributes on a frame. Returns list of InsightFace Face objects.
        """
        try:
            return self.app.get(frame)
        except Exception:
            return []

    def assign_genders(
        self,
        tracks: Dict[int, dict],
        faces: List,
        track_gender: Dict[int, str],
        track_gender_conf: Dict[int, float],
    ) -> None:
        """
        Assign genders to tracks by matching detected faces to person boxes.
        Updates track_gender and track_gender_conf in place.
        """
        # Preprocess faces: (face_center, gender, gconf)
        processed = []
        for f in faces:
            fb = getattr(f, "bbox", None)
            if fb is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in fb]
            fctr = ((x1 + x2) // 2, (y1 + y2) // 2)
            g, gconf = _map_gender(f)
            processed.append((fctr, g, gconf))

        # For each track, find nearest face center that lies inside its bbox
        for tid, tr in tracks.items():
            best = None
            best_dist = 1e9
            tctr = center_of_box(tr["xyxy"])
            for fctr, g, gconf in processed:
                if point_in_box(fctr, tr["xyxy"]):
                    d = distance(tctr, fctr)
                    if d < best_dist:
                        best = (g, gconf)
                        best_dist = d
            if best is not None:
                g, gconf = best
                if g != "U" and (gconf >= track_gender_conf[tid] or track_gender[tid] == "U"):
                    track_gender[tid] = g
                    track_gender_conf[tid] = gconf