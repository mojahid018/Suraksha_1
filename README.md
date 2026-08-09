# 🛡️ Suraksha — Women Safety Real-Time Analytics

> **An AI-powered real-time video analytics system designed to identify potentially risky situations by detecting, tracking, and analyzing people in surveillance footage.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-purple)
![InsightFace](https://img.shields.io/badge/InsightFace-Face%20Analysis-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![ByteTrack](https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

**Suraksha** is an AI-based real-time surveillance analytics application focused on identifying potentially unsafe situations in public or monitored environments.

The system processes a live camera feed and combines:

* **YOLOv8** for person detection
* **ByteTrack** for persistent multi-person tracking
* **InsightFace** for face analysis and gender estimation
* **OpenCV** for video processing and visualization
* **Rule-based risk analysis** for identifying potentially suspicious situations
* **Flask** for serving the real-time video stream through a web interface

The system continuously analyzes the scene and displays:

* Number of detected men and women
* Person tracking IDs
* Estimated gender labels
* Male-to-female ratio
* Current risk level
* Risk score
* Detected risk events
* Real-time FPS

---

## ✨ Key Features

### 🎥 Real-Time Video Analytics

Processes a live webcam stream and performs computer vision analysis frame by frame.

### 👤 Person Detection

Uses **YOLOv8** to detect people in the video stream.

### 🎯 Multi-Object Tracking

Uses **ByteTrack** to maintain consistent tracking IDs across frames.

Example:

```text
ID 1 (M)
ID 2 (F)
ID 3 (M)
```

### 🧑 Gender Estimation

Uses **InsightFace** to analyze detected faces and estimate gender.

Each tracked person is assigned:

```text
M → Male
F → Female
U → Unknown
```

### ⚠️ Risk Detection

The system currently evaluates multiple potentially risky situations:

* A woman surrounded by multiple men
* A man rapidly approaching a woman
* A potentially fallen/lying person

### 🚨 Male-to-Female Ratio Alert

A configurable alert system monitors the male-to-female ratio.

The default threshold is:

```text
Male : Female ≥ 3 : 1
```

Alerts include a cooldown mechanism to avoid repeatedly triggering alerts every frame.

### 📊 Real-Time Risk Score

Risk events contribute different scores to produce an overall risk level:

| Risk Score | Level     |
| ---------: | --------- |
|        0–1 | 🟢 LOW    |
|        2–4 | 🟠 MEDIUM |
|         5+ | 🔴 HIGH   |

### 🖥️ Live Monitoring Dashboard

The Flask web interface displays the processed video stream with an analytics overlay.

---

# 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │   Webcam / Camera   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       YOLOv8        │
                    │  Person Detection   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ByteTrack      │
                    │  Person Tracking    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     InsightFace     │
                    │   Face / Gender     │
                    │      Analysis       │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌────────────┐ ┌─────────────┐ ┌─────────────┐
        │   Gender   │ │   Movement  │ │   Spatial   │
        │   Counts   │ │   Analysis  │ │  Relations  │
        └──────┬─────┘ └──────┬──────┘ └──────┬──────┘
               │              │               │
               └──────────────┼───────────────┘
                              ▼
                    ┌─────────────────────┐
                    │    Risk Engine      │
                    │ Rule-Based Analysis │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             ┌────────────┐       ┌────────────┐
             │ Risk Score │       │   Alerts   │
             └──────┬─────┘       └────────────┘
                    │
                    ▼
             ┌────────────────┐
             │ OpenCV Overlay │
             └───────┬────────┘
                     │
                     ▼
             ┌────────────────┐
             │ Flask Web App  │
             └────────────────┘
```

---

# 🔬 Detection Pipeline

The application follows a multi-stage computer vision pipeline.

```text
Camera Frame
     ↓
YOLOv8 Person Detection
     ↓
ByteTrack Tracking
     ↓
Track History
     ↓
InsightFace Face Analysis
     ↓
Gender Assignment
     ↓
Risk Event Detection
     ↓
Risk Score Calculation
     ↓
Visualization Overlay
     ↓
Flask Video Stream
```

---

# 🤖 AI & Computer Vision Components

## 1. YOLOv8 — Person Detection

The system uses the **YOLOv8 Nano model (`yolov8n.pt`)** for lightweight real-time person detection.

Only the COCO `person` class is retained:

```text
Class ID: 0
Class: person
```

Detection parameters currently include:

```text
Confidence threshold: 0.35
IoU threshold:         0.45
```

---

## 2. ByteTrack — Multi-Object Tracking

YOLOv8's tracking functionality is used with:

```text
bytetrack.yaml
```

This allows the system to maintain persistent IDs for people across frames.

For example:

```text
Frame 1 → Person ID 4
Frame 2 → Person ID 4
Frame 3 → Person ID 4
```

This tracking history is important for detecting movement patterns such as rapid approach.

---

## 3. InsightFace — Gender Estimation

The project uses:

```text
InsightFace
Model: buffalo_l
Provider: CPUExecutionProvider
```

Face detections are matched to YOLO person bounding boxes.

The system then maintains a gender state for each track:

```text
Track ID → Gender → Confidence
```

The strongest observed gender confidence is retained for the tracked person.

---

# ⚠️ Risk Analysis

The risk engine is implemented in:

```text
wsafety/risk.py
```

It currently evaluates three major patterns.

---

## 1. Woman Surrounded by Multiple Men

The system identifies a potentially concerning situation when:

```text
Female nearby males >= 2
AND
No nearby females
```

This contributes:

```text
Risk Score +3
```

Example:

```text
       👨
        \
         👩
        /
       👨

Potential Event:
Female surrounded by multiple males
```

---

## 2. Rapid Approach

The system maintains a short movement history for each tracked person.

It compares:

```text
Previous distance
        ↓
Current distance
```

If the distance between a male and female decreases significantly while they are already close to each other, the system generates a potential rapid-approach event.

Risk contribution:

```text
Risk Score +2
```

---

## 3. Possible Fallen Person

The system analyzes the aspect ratio of a person's bounding box.

A person may be flagged as potentially lying/fallen when:

```text
Height / Width < 0.55
```

and the bounding box is sufficiently large relative to the frame.

Risk contribution:

```text
Risk Score +3
```

---

# 🚨 Risk Classification

Risk levels are calculated from the cumulative event score.

```text
Score 0–1
    ↓
LOW

Score 2–4
    ↓
MEDIUM

Score 5+
    ↓
HIGH
```

The current risk state is displayed directly on the video stream.

Example:

```text
RISK: HIGH (score=5)
```

---

# 🔔 Ratio Alert System

The project contains a dedicated alert component:

```text
wsafety/alert.py
```

The default threshold is:

```text
Male / Female >= 3.0
```

For example:

```text
Men = 6
Women = 2

Ratio = 6 / 2
      = 3.0

→ ALERT
```

### Cooldown

To prevent repeated alerts from being printed continuously, the system uses a default cooldown of:

```text
10 seconds
```

The alert system also handles the case where there are no detected females according to its configured behavior.

---

# 📊 Real-Time Visualization

OpenCV overlays analytics directly onto the camera frame.

The visualization contains:

### Person Tracking

```text
ID 1 (M)
ID 2 (F)
ID 3 (U)
```

### Population Statistics

```text
Men 3
Women 1
Ratio 3:1
```

### Risk Status

```text
RISK: HIGH (score=5)
```

### Detected Events

```text
Female 2 surrounded by 2 males
Male 4 rapidly approaching Female 2
```

### Performance

```text
28.4 FPS
```

---

# 🗂️ Project Structure

```text
Suraksha_1-main/
│
├── app.py
├── requirements.txt
├── yolov8n.pt
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── wsafety/
    ├── __init__.py
    ├── alert.py
    ├── detector.py
    ├── gender.py
    ├── risk.py
    ├── utils.py
    └── viz.py
```

---

## 📄 File Responsibilities

### `app.py`

Main Flask application.

Responsible for:

* Starting Flask
* Initializing detection components
* Capturing webcam frames
* Running the complete analytics pipeline
* Streaming processed frames

---

### `wsafety/detector.py`

Contains the `PersonDetector` class.

Responsible for:

* Loading YOLOv8
* Person detection
* ByteTrack integration
* Maintaining tracking IDs
* Extracting bounding boxes and confidence scores

---

### `wsafety/gender.py`

Contains the `GenderEstimator` class.

Responsible for:

* InsightFace initialization
* Face detection
* Gender estimation
* Matching faces to tracked persons
* Maintaining gender confidence

---

### `wsafety/risk.py`

Contains the risk-analysis engine.

Responsible for:

* Detecting risky spatial relationships
* Detecting rapid approaches
* Detecting potentially fallen people
* Calculating risk scores
* Converting scores into LOW/MEDIUM/HIGH levels

---

### `wsafety/alert.py`

Contains the `RatioAlert` class.

Responsible for:

* Male-to-female ratio monitoring
* Threshold-based alerts
* Alert cooldown
* Preventing repeated alerts

---

### `wsafety/viz.py`

Responsible for drawing analytics on video frames.

Includes:

* Bounding boxes
* Person IDs
* Gender labels
* Men/women counts
* Gender ratio
* Risk badge
* Event panel
* FPS indicator

---

### `wsafety/utils.py`

Contains reusable computer-vision utilities such as:

* Bounding-box center calculation
* Bounding-box dimensions
* Bounding-box diagonal
* Point-in-box testing
* Euclidean distance
* IoU calculation

---

# 🛠️ Tech Stack

## Programming

* Python

## Computer Vision

* OpenCV
* InsightFace

## Deep Learning

* YOLOv8
* ONNX Runtime
* PyTorch

## Object Tracking

* ByteTrack

## Web Framework

* Flask
* Jinja2

## Data / Scientific Computing

* NumPy
* SciPy
* Pandas
* Scikit-learn

---

# ⚙️ Installation

## Prerequisites

Recommended environment:

```text
Python 3.10+
Webcam
Internet connection for initial model/package setup
```

A GPU is recommended for better real-time performance, although the current InsightFace configuration explicitly uses the CPU execution provider.

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/suraksha.git

cd suraksha
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses a relatively large computer-vision stack, including:

```text
ultralytics
insightface
onnxruntime
opencv
torch
torchvision
Flask
```

---

# ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

The application runs on:

```text
http://localhost:5000
```

Open the address in a browser and allow camera access if prompted.

---

# 🎥 Camera Configuration

The current implementation uses the default webcam:

```python
source=0
```

If you have multiple cameras, the source can be changed.

For example:

```python
source=1
```

For video files or other supported sources, the detector can also accept a different source.

---

# 🔧 Configuration

## Detection Configuration

Current defaults:

```text
YOLO Model:       yolov8n.pt
Confidence:       0.35
IoU:              0.45
Tracker:          bytetrack.yaml
```

---

## Gender Analysis

Current configuration:

```text
InsightFace Model: buffalo_l
Execution Provider: CPUExecutionProvider
Detection Size: 640 × 640
```

---

## Alert Configuration

Default:

```text
Threshold: 3.0
Cooldown: 10 seconds
Require Female: True
```

These values can be modified in:

```text
app.py
```

---

# 📈 Performance

Performance depends heavily on:

* CPU/GPU
* Camera resolution
* Number of people in the scene
* YOLO model size
* InsightFace processing frequency
* Lighting conditions

To reduce computation, face analysis is cached and refreshed every **5 frames** in the current implementation.

```python
if frame_idx % 5 == 0:
    faces_cache[:] = gender_est.get_faces(frame)
```

The processed stream also displays real-time FPS.

---

# 🧪 Example Scenario

Suppose a camera detects:

```text
3 Men
1 Woman
```

and two men are in close proximity to the woman.

The system may generate:

```text
Men 3
Women 1
Ratio 3:1

RISK: HIGH
```

and potentially report:

```text
Female 7 surrounded by 2 males in close proximity
```

The system then displays the event directly on the video overlay and prints ratio alerts to the terminal when the configured threshold is reached.

---

# 🎯 Project Objectives

The main objectives of Suraksha are:

1. Apply AI and computer vision to a real-world safety problem.
2. Detect and track people in real time.
3. Analyze spatial relationships between tracked individuals.
4. Identify potentially risky movement patterns.
5. Provide visual risk indicators to a monitoring interface.
6. Demonstrate the integration of multiple AI models into a single pipeline.

---

# 💡 Key Learning Outcomes

This project provided practical experience with:

* Real-time computer vision
* Object detection
* Multi-object tracking
* YOLOv8
* ByteTrack
* Face analysis
* Computer vision heuristics
* Spatial-distance analysis
* Video-stream processing
* OpenCV visualization
* Flask streaming
* Model integration
* Real-time performance optimization
* Modular Python architecture

---

# 🔐 Privacy & Ethical Considerations

Suraksha is intended as a **prototype for safety-oriented video analytics**.

The system performs automated analysis of people in video footage and includes gender estimation. Such technology can produce inaccurate classifications and should **not be treated as a definitive judgment of a person's gender, behavior, intent, or threat level**.

Potential real-world deployments should consider:

* User consent
* Data protection
* Secure video handling
* Bias and model accuracy
* False positives and false negatives
* Human review before taking action
* Applicable privacy and surveillance regulations

The risk score should be interpreted as a **heuristic indicator**, not proof that an incident is occurring.

---

# ⚠️ Limitations

The current prototype has several limitations:

* Risk rules are heuristic rather than learned from labeled safety incidents.
* Gender estimation can be incorrect, especially with poor visibility.
* Face detection may fail under occlusion or unfavorable lighting.
* Risk detection depends on camera angle and scene geometry.
* A fallen-person heuristic can generate false positives.
* A high male-to-female ratio does not inherently indicate danger.
* Current alerts are terminal-based rather than connected to an external notification service.
* The application currently uses a local webcam source.
* Performance may decrease with crowded scenes or limited hardware.

---

# 🔮 Future Improvements

Potential future enhancements include:

* [ ] Add configurable video/camera sources through the UI
* [ ] Add RTSP/IP camera support
* [ ] Add SMS/email/mobile notifications
* [ ] Add emergency alert integration
* [ ] Add incident recording
* [ ] Add event timestamps
* [ ] Add dashboard analytics
* [ ] Add historical risk reports
* [ ] Add configurable risk thresholds
* [ ] Add more robust pose estimation
* [ ] Add suspicious-behavior classification
* [ ] Add crowd-density analysis
* [ ] Add zone-based monitoring
* [ ] Add geofenced safety zones
* [ ] Add database-backed event storage
* [ ] Add authentication and role-based access
* [ ] Add GPU acceleration configuration
* [ ] Improve false-positive handling
* [ ] Add automated testing
* [ ] Deploy as a production-ready monitoring platform

---

# 🤝 Contributing

Contributions are welcome.

### 1. Fork the repository

### 2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

### 3. Make your changes

### 4. Commit

```bash
git commit -m "Add new feature"
```

### 5. Push

```bash
git push origin feature/new-feature
```

### 6. Open a Pull Request

---

# 📜 License

This project is released under the **MIT License**.

---

# 👨‍💻 Author

**Mojahid Ansari**

Computer Science Engineering | AI/ML & Data Science

### Areas of Interest

* 🤖 Artificial Intelligence
* 👁️ Computer Vision
* 🧠 Machine Learning
* 📊 Data Science
* 🐍 Python
* 🚀 AI-powered Applications

---

# ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

## 🙏 Acknowledgements

This project builds upon several open-source technologies and research ecosystems:

* YOLO / Ultralytics
* ByteTrack
* InsightFace
* OpenCV
* Flask
* PyTorch
* ONNX Runtime

---

> **Suraksha combines real-time object detection, multi-object tracking, face analysis, and rule-based risk assessment to explore how AI can support safety-oriented video monitoring.**
