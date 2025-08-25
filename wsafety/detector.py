from typing import Dict, Generator, Optional

import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """
    Wrapper around YOLOv8 with built-in ByteTrack. Use track_stream to iterate frames.
    After each yield, current_tracks contains dict of {track_id: {"xyxy": [x1,y1,x2,y2], "conf": float}}
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)
        self.current_tracks: Dict[int, dict] = {}

    def _parse_tracks_from_result(self, result) -> Dict[int, dict]:
        self.current_tracks = {}
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return self.current_tracks

        xyxy_all = boxes.xyxy.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy_all),))
        cls_all = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(xyxy_all),), dtype=int)
        ids_all = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy_all))

        # Keep COCO class 0: 'person'
        for i in range(len(xyxy_all)):
            if cls_all[i] != 0:
                continue
            tid = int(ids_all[i])
            self.current_tracks[tid] = {
                "xyxy": xyxy_all[i].tolist(),
                "conf": float(conf_all[i]),
            }
        return self.current_tracks

    def track_stream(
        self,
        source: str,
        conf: float = 0.35,
        iou: float = 0.45,
        tracker: str = "bytetrack.yaml",
    ) -> Generator[Optional[np.ndarray], None, None]:
        """
        Yields BGR frames with tracking state available in self.current_tracks.
        Convert numeric string sources like "0" to int camera index.
        """
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        results_gen = self.model.track(
            source=source,
            stream=True,
            conf=conf,
            iou=iou,
            tracker=tracker,
            persist=True,
            verbose=False,
        )

        for result in results_gen:
            frame = getattr(result, "orig_img", None)
            if frame is None:
                yield None
                continue

            self._parse_tracks_from_result(result)
            yield frame