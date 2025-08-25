
from flask import Flask, render_template, Response
import cv2
from wsafety.detector import PersonDetector
from wsafety.gender import GenderEstimator
from wsafety.risk import compute_risk_events, risk_level
from wsafety.utils import center_of_box
from wsafety.viz import draw_frame
from wsafety.alert import RatioAlert
from collections import defaultdict, deque
import time

app = Flask(__name__)

# Initialize components
detector = PersonDetector(model_name="yolov8n.pt")
gender_est = GenderEstimator(providers=["CPUExecutionProvider"])
ratio_alert = RatioAlert(threshold=3.0, cooldown_seconds=10.0, require_female=True)

# State
track_gender = defaultdict(lambda: "U")
track_gender_conf = defaultdict(float)
track_history = defaultdict(lambda: deque(maxlen=12))
faces_cache = []

frame_idx = 0
t_prev = time.time()

def generate_frames():
    global frame_idx, t_prev
    for frame in detector.track_stream(source=0, conf=0.35, iou=0.45, tracker="bytetrack.yaml"):
        if frame is None:
            continue

        H, W = frame.shape[:2]
        tracks = detector.current_tracks

        for tid, tr in tracks.items():
            track_history[tid].append(center_of_box(tr["xyxy"]))

        if frame_idx % 5 == 0:
            faces_cache[:] = gender_est.get_faces(frame)
        gender_est.assign_genders(tracks, faces_cache, track_gender, track_gender_conf)

        male_count = sum(1 for tid in tracks if track_gender[tid] == "M")
        female_count = sum(1 for tid in tracks if track_gender[tid] == "F")
        events, score = compute_risk_events(tracks, track_gender, track_history, frame.shape)
        level = risk_level(score)

        ratio_alert.update(male_count, female_count)

        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        t_prev = t_now
        fps = 1.0 / dt
        t_prev = t_now

        frame_vis = draw_frame(frame, tracks, track_gender, male_count, female_count, events, level, score, fps)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame_vis)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_idx += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
