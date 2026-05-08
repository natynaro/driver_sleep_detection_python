# Driver Sleep Detection – Python, MediaPipe & PyQt5 (Evaluable System)

This project implements an **evaluable and comparable real-time driver drowsiness detection system** using physiologically meaningful rules. It compares **Visual-only** vs. **Visual + Wearable** models through quantitative metrics, designed for academic evaluation and safety analysis.

## 🚀 Key Features

- **Physiologically Realistic Detection**: Based on EAR (Eye Aspect Ratio), microsleep, yawning, and head tilt – no arbitrary scoring.
- **Model Comparison**: Evaluates Visual-only vs. Visual+Wearable fusion with metrics for quality, resources, and temporal evolution.
- **Conservative Fusion Logic**: Prioritizes safety; SLEEPY visual forces SLEEPY fused to avoid false negatives.
- **Real-time Metrics**: CPU usage, detection delays, coincidences, energy estimates.
- **In-window Alert**: Color-coded warning button for DROWSY and MICROSLEEP states.
- **Data Export**: Automatic CSV logging for post-analysis and reporting.
- **PyQt5 GUI**: Enhanced interface showing all models and metrics in real-time.

---

## 📁 Project Structure

```
driver_sleep_detection_python/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── models/
│   └── face_landmarker.task              # MediaPipe Face Landmarker model
├── data/
│   ├── wearable_ble_stream.xlsx          # Simulated wearable BLE data (Excel)
│   └── session_logs.csv                  # Auto-exported metrics (generated)
└── src/
    ├── __init__.py
    ├── camera/
    │   ├── __init__.py
    │   ├── webcam.py                      # Webcam capture wrapper
    │   └── resolution_test.py             # Camera resolution testing utility
    ├── detection/
    │   ├── __init__.py
    │   ├── eye_aspect_ratio.py            # EAR calculation module
    │   └── face_mesh_detector.py          # MediaPipe face detection wrapper
    ├── gui/
    │   ├── __init__.py
    │   └── qt_app.py                      # Main PyQt5 application with evaluation
    └── metrics/
        ├── __init__.py
        └── metrics_collector.py            # Metrics collection and export module
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Webcam connected
- Excel file: `data/wearable_ble_stream.xlsx` (simulated wearable data)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` – Video capture and processing
- `mediapipe` – Face landmark detection
- `numpy` – Numerical computations
- `PyQt5` – GUI framework
- `pandas` – Excel reading and data export

### Download the Face Landmarker Model

Place the MediaPipe model in `models/face_landmarker.task`:
1. Download from: https://storage.googleapis.com/mediapipe-tasks/python/latest/face_landmarker.task
2. Place in: `models/face_landmarker.task`

---

## 🎯 How It Works

### Core Architecture

The system operates in four layers for evaluation:

#### 1. **Face Detection & Landmark Extraction** (`detection/face_mesh_detector.py`)

Uses MediaPipe FaceLandmarker to detect 468 facial landmarks:
- **Key landmarks**:
  - Eyes: 159,145,153,154,155,133 (left); 386,374,380,381,382,362 (right)
  - Mouth: 13 (top), 14 (bottom) for yawn
  - Head: 1 (nose), 152 (chin) for tilt

#### 2. **Physiological Feature Calculation** (`detection/eye_aspect_ratio.py`)

**Eye Aspect Ratio (EAR):**
$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}$$

**Yawn Ratio:** Normalized mouth opening vs. face height.

**Head Tilt Angle:** Inclination from nose-chin vector.

#### 3. **State Machines & Fusion** (`gui/qt_app.py`)

**Visual Model States:**
- **AWAKE**: Normal behavior
- **DROWSY**: Yawn + head tilt in window, or sustained head tilt
- **SLEEPY**: Microsleep (eyes closed > ~0.25s)

**Wearable Model:**
- **DROWSY** if `drowsiness_level > 0.6`
- Data from Excel, interpolated temporally

**Fusion Model (Conservative):**
- If Visual == SLEEPY → Fused = SLEEPY
- If Visual == DROWSY OR Wearable == DROWSY → Fused = DROWSY
- Else → Fused = AWAKE

#### 4. **Metrics Collection** (`metrics/metrics_collector.py`)

**Quality Metrics:**
- **Detection Delay**: Time from Wearable DROWSY to Visual/Fused DROWSY
- **Coincidence %**: Frame-by-frame agreement between Visual and Wearable

**Resource Metrics:**
- **CPU %**: Average usage (psutil)
- **Energy Estimate**: Cumulative CPU-seconds (logical units)

**Temporal Evolution:**
- Timeline of state changes
- CSV export: timestamp, visual_state, wearable_drowsy, fused_state, cpu_percent, energy_estimate

---

## ⚙️ Configuration Parameters

### Detection Thresholds (Physiologically Based)

```python
# Camera & Processing
MOBILE_RESOLUTION = (640, 480)
FPS = 30.0
DT = 1.0 / FPS

# EAR (Eye Aspect Ratio)
EYE_OPEN_THRESHOLD = 0.75      # Baseline for open eyes
EYE_CLOSED_RATIO = 0.92        # Closed eye ratio vs. baseline
MICROSLEEP_FRAMES = 8          # ~0.25s at 30 FPS

# Facial Cues
YAWN_RATIO_THRESHOLD = 0.12    # Normalized mouth opening
HEAD_TILT_THRESHOLD = 0.15     # Head inclination angle

# Temporal Logic (seconds)
YAWN_TILT_WINDOW = 2.0         # Window for yawn + tilt → DROWSY
HEAD_TILT_SUSTAINED = 2.0      # Sustained tilt → DROWSY
AWAKE_RESET_TIME = 4.0         # Normal behavior → AWAKE
```

### Wearable Configuration
- **DROWSY Threshold**: `drowsiness_level > 0.6`
- **Packet Loss Handling**: Uses last valid value if `packet_lost == 1`
- **Temporal Alignment**: `t_arrival_ms` converted to seconds, interpolated

---

## 💻 Usage

### Main Application (Evaluable GUI)

```bash
python -m src.gui.qt_app
```

**Real-time Display:**
- **Video Feed**: Live camera with facial overlays
- **States**:
  - Visual Model: AWAKE/DROWSY/SLEEPY
  - Wearable: AWAKE/DROWSY (with level)
  - Fused Model: AWAKE/DROWSY/SLEEPY (blue text)
- **Alert Button**: In-window warning button that turns orange for DROWSY and red for MICROSLEEP
- **Metrics**:
  - EAR: Current eye aspect ratio
  - Yawn: Yes/No
  - Head Tilt: Yes/No
  - Microsleep: Yes/No
  - CPU: Average %
  - Delay: Detection delay (s)
  - Coinc: Agreement % with wearable
- **Timeline**: Recent events (e.g., "Wearable DROWSY at 10.5s")

**Interaction:**
- Close eyes >0.25s → Microsleep → SLEEPY
- Yawn + tilt head → DROWSY
- Sustained head tilt → DROWSY
- Normal behavior → AWAKE after 4s

**Export**: Closes automatically saves `data/session_logs.csv`

### Analysis Script (Post-Evaluation)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load exported data
df = pd.read_csv('data/session_logs.csv')

# Calculate final metrics
delay_visual = df[df['wearable_drowsy']].index[0] - df[df['visual_state'] == 'DROWSY'].index[0] if any(df['visual_state'] == 'DROWSY') else None
coincidence = (df['visual_state'].isin(['DROWSY', 'SLEEPY']) == df['wearable_drowsy']).mean() * 100
cpu_avg = df['cpu_percent'].mean()
energy_total = df['energy_estimate'].iloc[-1]

# Plot evolution
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['cpu_percent'], label='CPU %')
plt.plot(df['timestamp'], df['energy_estimate'], label='Energy Estimate')
plt.legend()
plt.title('Resource Metrics Over Time')

plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['visual_state'].map({'AWAKE': 0, 'DROWSY': 1, 'SLEEPY': 2}), label='Visual')
plt.plot(df['timestamp'], df['wearable_drowsy'].astype(int), label='Wearable')
plt.plot(df['timestamp'], df['fused_state'].map({'AWAKE': 0, 'DROWSY': 1, 'SLEEPY': 2}), label='Fused')
plt.legend()
plt.title('State Evolution')
plt.show()
```

---

## 📊 Evaluation Framework

### Model Comparison
- **Visual-only**: Uses facial cues only
- **Visual+Wearable**: Fused model with Excel data
- **Trade-offs**: Precision vs. computational cost

### Key Metrics for Academic Reporting
- **Safety (Delay)**: How quickly each model detects drowsiness
- **Accuracy (Coincidence)**: Agreement between models
- **Efficiency (CPU/Energy)**: Resource usage for mobile deployment
- **Temporal Analysis**: Timeline shows when each model triggers

### Justification
- **Conservative Fusion**: Prevents false negatives in safety-critical system
- **Physiological Basis**: Thresholds based on human sleep research, not ML scoring
- **Evaluable Design**: Quantifiable metrics for thesis/defense

---

## 🔧 Troubleshooting

### Model Not Found
- Ensure `face_landmarker.task` is in `models/`

### Excel Not Found
- Place `wearable_ble_stream.xlsx` in `data/`

### Low Performance
- Reduce resolution in `MOBILE_RESOLUTION`
- Close background apps

### False Positives
- Adjust thresholds: Increase `EYE_CLOSED_RATIO` or `MICROSLEEP_FRAMES`

### Face Not Detected
- Improve lighting, face camera directly

---

## 📝 Academic References

- MediaPipe: https://mediapipe.dev/
- EAR Algorithm: Soukupová & Čech (2016)
- Drowsiness Detection: Research on physiological markers

**For Thesis/Report**: This system enables comparative analysis of multimodal drowsiness detection, justifying design choices with empirical metrics.
- **PyQt5:** https://www.riverbankcomputing.com/software/pyqt/

---

## 📌 Notes

- This system is designed for **driver safety research and testing**
- **Accuracy varies** based on lighting, camera quality, and user positioning
- **Always use in combination with other safety systems**
- Real deployment should include additional validation and calibration

---

**Last Updated:** May 2026
**Python Version:** 3.10+
**Status:** Development Complete
