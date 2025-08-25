# Women Safety Analytics (Modular)

Detect people in live/video streams, estimate M:F ratios, and flag potentially risky situations using simple heuristics.

Modules
- wsafety/detector.py — YOLOv8 person detection + ByteTrack streaming
- wsafety/gender.py — Face-based gender estimation (InsightFace)
- wsafety/risk.py — Heuristic risk detection (surrounded, rapid approach, fallen)
- wsafety/utils.py — Geometry helpers
- wsafety/viz.py — Drawing overlays
- app.py — Orchestrates everything

Quick start
1) Create and activate a virtual environment
   - Windows:
     - py -m venv .venv
     - .venv\Scripts\activate
   - macOS/Linux:
     - python3 -m venv .venv
     - source .venv/bin/activate

2) Install dependencies
   - python -m pip install -U pip setuptools wheel
   - python -m pip install -r requirements.txt

3) Run
   - Webcam: python app.py --source 0
   - Video:  python app.py --source "path/to/video.mp4"
   - Stream: python app.py --source "rtsp://user:pass@ip/..."

4) Options
   - --save out.mp4 to save annotated video
   - --face_every_n 8 to reduce CPU usage
   - --model yolov8s.pt for higher accuracy (if your machine can handle it)

Notes and ethics
- First run downloads YOLOv8 weights and InsightFace models; allow 1–2 minutes.
- Gender estimation can be inaccurate and biased. Use it only as supportive signal.
- Always follow local laws and privacy rules for surveillance.

Troubleshooting
- Camera black: close other apps, try --source 1, grant camera permissions.
- macOS Apple Silicon: onnxruntime-silicon is selected automatically by requirements.
- Linux libGL error: sudo apt-get install -y libgl1