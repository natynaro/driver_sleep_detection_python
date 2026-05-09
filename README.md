# Driver Sleep Detection – Python, MediaPipe & PyQt5 (Evaluable System)

This project implements an **evaluable and comparable real-time driver drowsiness detection system** using physiologically meaningful rules and multimodal evaluation. The system supports both:

* **Visual-only mode**
* **Visual + Wearable fusion mode**

The application measures not only drowsiness states, but also **detection quality, computational cost, temporal evolution, and experimental configuration impact**.

---

# 🚀 Key Features

* **Physiologically Realistic Detection**

  * EAR (Eye Aspect Ratio)
  * Microsleep detection
  * Yawning analysis
  * Head tilt analysis

* **Two Experimental Modes**

  * `VISUAL_ONLY`
  * `FUSED`

* **Conservative Fusion Logic**

  * Prevents false negatives in safety-critical situations
  * Visual microsleep always dominates fusion state

* **Real-Time Metrics**

  * CPU usage
  * Detection delay
  * Coincidence percentage
  * Energy estimation

* **Temporal Event Timeline**

  * Real-time logging of state transitions

* **Automatic CSV Export**

  * Session metrics exported for academic analysis

* **PyQt5 GUI**

  * Live video feed
  * State visualization
  * Alert system
  * Metrics dashboard

* **Experimental Camera Configuration**

  * Resolution comparison
  * FPS comparison
  * Performance evaluation

---

# 📁 Project Structure

```text
driver_sleep_detection_python/
├── README.md
├── requirements.txt
├── models/
│   └── face_landmarker.task
├── data/
│   ├── wearable_ble_stream.xlsx
│   └── session_logs.csv
└── src/
    ├── __init__.py
    ├── camera/
    │   ├── __init__.py
    │   ├── webcam.py
    │   └── resolution_test.py
    ├── detection/
    │   ├── __init__.py
    │   ├── eye_aspect_ratio.py
    │   └── face_mesh_detector.py
    ├── gui/
    │   ├── __init__.py
    │   └── qt_app.py
    └── metrics/
        ├── __init__.py
        └── metrics_collector.py
```

---

# 🛠️ Installation & Setup

## Prerequisites

* Python 3.10+
* Webcam
* Excel wearable data file:

  * `data/wearable_ble_stream.xlsx`

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Dependencies

* `opencv-python`
* `mediapipe`
* `numpy`
* `pandas`
* `PyQt5`
* `psutil`
* `openpyxl`

---

## Download MediaPipe Face Landmarker Model

Download:

```text
https://storage.googleapis.com/mediapipe-tasks/python/latest/face_landmarker.task
```

Place the file inside:

```text
models/face_landmarker.task
```

---

# 🎯 System Architecture

The system is divided into four main layers:

---

## 1. Face Detection & Landmark Extraction

**File:** `detection/face_mesh_detector.py`

Uses MediaPipe FaceLandmarker to detect facial landmarks in real time.

### Main Landmarks Used

| Feature   | Landmark IDs            |
| --------- | ----------------------- |
| Left Eye  | 159,145,153,154,155,133 |
| Right Eye | 386,374,380,381,382,362 |
| Mouth     | 13, 14                  |
| Head Tilt | 1 (nose), 152 (chin)    |

---

## 2. Physiological Feature Extraction

**File:** `detection/eye_aspect_ratio.py`

### Eye Aspect Ratio (EAR)

```math
EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}
```

### Additional Features

* Microsleep detection
* Mouth opening ratio
* Head inclination angle

---

## 3. Driver State Machine

**File:** `gui/qt_app.py`

### Driver States

* `AWAKE`
* `DROWSY`
* `SLEEPY`

### State Logic

#### Rule 1 — Microsleep

```text
Eyes closed for ~0.25s → SLEEPY
```

#### Rule 2 — Yawn + Head Tilt

```text
Yawn + tilt within temporal window → DROWSY
```

#### Rule 3 — Sustained Head Tilt

```text
Sustained head tilt → DROWSY
```

#### Rule 4 — Recovery

```text
Stable normal behavior → AWAKE
```

---

# 🔀 Experimental Modes

The system now supports configurable experimental modes:

```python
EXPERIMENT_MODE = "VISUAL_ONLY"
# Options:
# "VISUAL_ONLY"
# "FUSED"
```

---

## VISUAL_ONLY Mode

Uses only visual facial features:

* EAR
* Microsleep
* Yawning
* Head tilt

No wearable data is processed.

---

## FUSED Mode

Combines:

* Visual model
* Wearable drowsiness stream

### Wearable Logic

```python
wearable_drowsy = wearable_level > 0.6
```

### Packet Loss Handling

If:

```python
packet_lost == 1
```

The system reuses the last valid wearable value.

### Temporal Alignment

Wearable timestamps are interpolated:

```python
np.interp(...)
```

---

# 🧠 Conservative Fusion Logic

The fusion system prioritizes safety:

```python
if visual_state == SLEEPY:
    fused = SLEEPY

elif visual_state == DROWSY or wearable_drowsy:
    fused = DROWSY

else:
    fused = AWAKE
```

This minimizes dangerous false negatives.

---

# 📊 Metrics Collection System

**File:** `metrics/metrics_collector.py`

The system includes a complete evaluation framework.

---

## Quality Metrics

### Detection Delay

Measures time between:

```text
Wearable DROWSY
→
Visual/Fused detection
```

---

### Coincidence Percentage

Frame-by-frame agreement between:

* Visual model
* Wearable model

---

## Resource Metrics

### CPU Usage

Measured using:

```python
psutil.cpu_percent()
```

---

### Energy Estimation

Simple cumulative CPU-based estimate:

```python
energy += cpu_percent / 100
```

---

## Temporal Evolution

Tracks:

* Visual state
* Wearable state
* Fused state
* CPU usage
* Energy estimation

---

# ⚙️ Configuration Parameters

## Camera Configuration

```python
CAMERA_RESOLUTION = (640, 480)
FPS = 30.0
DT = 1.0 / FPS
```

Alternative experiments:

```python
(320, 240)
FPS = 20.0
```

---

## EAR Configuration

```python
EYE_OPEN_THRESHOLD = 0.75
EYE_CLOSED_RATIO = 0.92
MICROSLEEP_FRAMES = 8
```

---

## Facial Cue Thresholds

```python
YAWN_RATIO_THRESHOLD = 0.12
HEAD_TILT_THRESHOLD = 0.15
```

---

## Temporal Logic

```python
YAWN_TILT_WINDOW = 2.0
HEAD_TILT_SUSTAINED = 2.0
AWAKE_RESET_TIME = 4.0
```

---

# 💻 Usage

## Run the Main Application

```bash
python -m src.gui.qt_app
```

---

# 🖥️ Real-Time GUI Features

## Live Video Feed

Displays:

* Face landmarks
* Driver monitoring overlay

---

## Driver States

### Visual Model

```text
AWAKE / DROWSY / SLEEPY
```

### Wearable State

```text
AWAKE / DROWSY
```

### Fused Model

```text
AWAKE / DROWSY / SLEEPY
```

---

## Metrics Dashboard

Displays:

* EAR
* Yawn detection
* Head tilt detection
* Microsleep detection
* CPU average
* Detection delay
* Coincidence percentage

---

## Timeline Viewer

Shows recent events:

```text
10.5s: Visual=DROWSY, Wearable=DROWSY, Fused=DROWSY
```

---

## Driver Warning System

### Normal

Green button:

```text
DRIVER STATUS: NORMAL
```

### Drowsy

Orange warning:

```text
WARNING: DROWSY DRIVER
```

### Microsleep

Red warning:

```text
WARNING: MICROSLEEP DETECTED
```

---

# 📁 CSV Export

When the application closes:

```python
self.metrics.export_to_csv('data/session_logs.csv')
```

The exported CSV contains:

| Column          | Description           |
| --------------- | --------------------- |
| timestamp       | Session time          |
| visual_state    | Visual model state    |
| wearable_drowsy | Wearable binary state |
| fused_state     | Fused model state     |
| cpu_percent     | CPU usage             |
| energy_estimate | Cumulative energy     |

---

# 📈 Example Post-Analysis

```python
import pandas as pd

df = pd.read_csv("data/session_logs.csv")

print(df.head())
print(df.describe())
```

Possible analyses:

* CPU evolution
* Detection latency
* Coincidence comparison
* Fusion performance
* Resolution/FPS impact

---

# 🧪 Experimental Evaluation

The system was designed for academic comparison experiments.

---

## Compared Configurations

### Detection Modes

* Visual-only
* Visual + wearable fusion

### Hardware Configurations

* 640×480 @ 30 FPS
* 320×240 @ 20 FPS

---

## Evaluation Goals

### Safety

How fast drowsiness is detected.

### Accuracy

Agreement between modalities.

### Efficiency

CPU and energy cost.

### Temporal Behavior

Evolution of driver states over time.

---

# 🔧 Troubleshooting

## Model Not Found

Ensure:

```text
models/face_landmarker.task
```

exists.

---

## Excel File Missing

Place:

```text
data/wearable_ble_stream.xlsx
```

inside the data folder.

---

## High CPU Usage

Try:

* Lower resolution
* Lower FPS
* Close background applications

---

## False Positives

Adjust:

```python
EYE_CLOSED_RATIO
MICROSLEEP_FRAMES
```

---

## Face Not Detected

Improve:

* Lighting
* Camera angle
* Face visibility

---

# 📝 Academic References

## MediaPipe

```text
https://mediapipe.dev/
```

---

## EAR Algorithm

Soukupová & Čech (2016)

---

## PyQt5

```text
https://www.riverbankcomputing.com/software/pyqt/
```

---

# 📌 Notes

* Designed for research and academic evaluation
* Real-world deployment requires calibration and validation
* Performance depends on lighting and camera quality
* Intended as a safety-support system, not a replacement for responsible driving

---

# ✅ Current Status

* Real-time detection implemented
* Experimental framework implemented
* Metrics system implemented
* CSV export implemented
* GUI completed
* Fusion logic completed

---

**Last Updated:** May 2026
**Python Version:** 3.10+
**Status:** Development Complete
