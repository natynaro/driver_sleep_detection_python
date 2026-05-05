# Driver Sleep Detection – Python, MediaPipe 0.10.x & PyQt5

This project implements a **real‑time driver drowsiness detection system** using:

- **MediaPipe Face Landmarker (0.10.x)**
- **OpenCV**
- **PyQt5**
- **Python 3.10+**

The application detects:

- **EAR (Eye Aspect Ratio)**
- **Blink count**
- **Blink duration**
- **Yawning**
- **Head tilt**
- **Real‑time FPS**
- **Overall driver alertness state**

A complete PyQt5 graphical interface is included.

---

##  Project Structure
```
driver_sleep_detection_python/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── models/
│   └── face_landmarker.task              # MediaPipe Face Landmarker model (0.10.x)
└── src/
    ├── __init__.py
    ├── app.py                             # Main OpenCV-based application
    ├── camera/
    │   ├── __init__.py
    │   ├── webcam.py                      # Webcam capture wrapper
    │   └── resolution_test.py             # Camera resolution testing utility
    ├── detection/
    │   ├── __init__.py
    │   ├── eye_aspect_ratio.py            # EAR calculation module
    │   └── face_mesh_detector.py          # MediaPipe face detection wrapper
    └── gui/
        ├── __init__.py
        └── qt_app.py                      # PyQt5 graphical interface
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Webcam connected to your system

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` – Video capture and image processing
- `mediapipe` – Face landmark detection
- `numpy` – Numerical computations
- `PyQt5` – GUI framework

### Download the Face Landmarker Model

The `face_landmarker.task` model must be downloaded and placed in the `models/` folder:

1. Download from: https://storage.googleapis.com/mediapipe-tasks/python/latest/face_landmarker.task
2. Place it in: `models/face_landmarker.task`

---

## 🎯 How It Works

### Core Architecture

The system operates in three main layers:

#### 1. **Face Detection & Landmark Extraction** (`detection/face_mesh_detector.py`)

Uses MediaPipe's FaceLandmarker (0.10.x) to detect 468 facial landmarks in real-time:

- **Input:** Video frame (RGB)
- **Output:** 468 normalized landmark points (x, y coordinates)
- **Key landmarks used:**
  - **Eyes:** Landmarks 159, 145, 153, 154, 155, 133 (left); 386, 374, 380, 381, 382, 362 (right)
  - **Mouth:** Landmarks 13 (top) and 14 (bottom) for yawn detection
  - **Head pose:** Landmarks 1 (nose) and 152 (chin) for tilt detection

```python
# Face detection example
frame, ear, face = detector.detect(frame)
if face is not None:
    # landmarks are available
```

#### 2. **Eye Aspect Ratio (EAR) Calculation** (`detection/eye_aspect_ratio.py`)

EAR measures eye openness using the Euclidean distance formula:

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}$$

Where p1-p6 are the 6 eye landmark points:
- p1, p4: Eye corners (left and right)
- p2, p3, p5, p6: Eye contour points

**Interpretation:**
- **EAR > 0.20 (threshold):** Eyes open → Awake state
- **EAR < 0.20:** Eyes closed → Potential sleep/blink

The average of left and right eyes is used for more robust detection.

#### 3. **Drowsiness Detection Logic** (`app.py` & `gui/qt_app.py`)

The system detects drowsiness by monitoring **consecutive frames** with closed eyes:

```
EAR < THRESHOLD for 15 consecutive frames (≈ 0.5 seconds at 30 FPS)
                    ↓
        SLEEP DETECTED! Alert triggered
```

**Thresholds & Parameters:**
- `SLEEP_THRESHOLD = 0.60` (app.py) or `0.20` (qt_app.py)
- `SLEEP_FRAMES_REQUIRED = 15` frames (~0.5 sec at 30 FPS)
- `YAWN_THRESHOLD = 25` pixels (mouth vertical distance)

### Detected Metrics

#### 1. **Blink Detection**
- Tracked when EAR transitions from low to high
- Records:
  - **Blink count:** Total number of blinks
  - **Blink duration:** Time eyes stayed closed (seconds)
  - **Last blink:** Duration of most recent blink

```python
if ear < SLEEP_THRESHOLD:
    if blink_start is None:
        blink_start = time.time()  # Start tracking
else:
    if blink_start is not None:
        blink_duration = time.time() - blink_start
        blink_durations.append(blink_duration)
        blink_count += 1
```

#### 2. **Yawn Detection**
- Measures vertical distance between top and bottom lip landmarks
- Threshold: mouth opening > 25 pixels
- Uses landmarks: 13 (top lip) and 14 (bottom lip)

```python
mouth_ratio = detector.mouth_open_ratio(face, w, h)
if mouth_ratio > YAWN_THRESHOLD:
    # Yawn detected
```

#### 3. **Head Tilt Detection**
- Calculates head inclination angle using nose and chin landmarks
- Computes: `angle = |Δx / Δy|` where Δx and Δy are horizontal/vertical distances
- Threshold: angle > 0.25

```python
nose = face.landmark[1]
chin = face.landmark[152]
angle = abs((x_chin - x_nose) / (y_chin - y_nose))
```

#### 4. **FPS (Frames Per Second)**
- Real-time performance metric
- Calculated: `FPS = 1 / (current_time - previous_time)`

#### 5. **Alert System**
- **Status display:** Shows "Awake" or "SLEEP DETECTED"
- **Visual indicators:** 
  - Red text = Sleep detected
  - Green text = Normal operation

---

## 💻 Usage

### Option 1: OpenCV Version (`src/app.py`)

**Fastest & most lightweight** – Uses only OpenCV for display

```bash
python -m src.app
```

**Controls:**
- Press **ESC** to exit

**Display Shows:**
- EAR value
- Real-time FPS
- Eye closure status
- Blink count and duration
- Yawn and head tilt detection
- Sleep alert status

### Option 2: PyQt5 GUI (`src/gui/qt_app.py`)

**Full-featured GUI** with controls and enhanced visualization

```bash
python -m src.gui.qt_app
```

**Features:**
- Start/Stop detection buttons
- Resolution selector (640x480, 1280x720, 1920x1080)
- Live video feed in window
- Color-coded status indicator
- All metrics displayed in real-time
- Clean, user-friendly interface

**GUI Controls:**
- **Start:** Begin drowsiness detection
- **Stop:** Pause detection
- **Resolution Dropdown:** Change camera resolution

### Mobile Simulation Mode

Both `src/app.py` and `src/gui/qt_app.py` now include a mobile simulation mode designed to mimic lower-end phone behavior.

**Enabled by default in the GUI:** `MOBILE_SIMULATION = True`

**What it simulates:**
- Lower camera resolution: `320x240`
- Reduced inference FPS: frame skipping via `SKIP_FRAMES = 2`
- Artificial CPU delay: `time.sleep(0.02)` after each inference
- Battery/CPU load simulation: `psutil.cpu_percent()` display
- Wearable/HR simulation: fake heart rate stream shown in the UI
- Mobile mode flag: `Mobile Mode: Yes` label

**Why it helps:**
- Better represents mobile ARM performance
- Shows how the app behaves under lower compute and camera constraints
- Useful for report comparison and stress testing

### Option 3: Resolution Testing (`src/camera/resolution_test.py`)

Test your camera's supported resolutions and FPS performance:

```bash
python -c "from src.camera.resolution_test import test_resolution; test_resolution(1920, 1080)"
```

This helps determine optimal settings for your hardware.

---

## 📊 Key Components Explained

### `Webcam` Class (`src/camera/webcam.py`)

Simple wrapper around OpenCV's VideoCapture:

```python
cam = Webcam(camera_id=0, width=640, height=480)
ret, frame = cam.read()
cam.release()
```

### `FaceMeshDetector` Class (`src/detection/face_mesh_detector.py`)

Core face detection using MediaPipe:

```python
detector = FaceMeshDetector(model_path="models/face_landmarker.task")
frame, ear, face = detector.detect(frame)

# Get additional metrics
angle = detector.head_tilt_angle(face, width, height)
mouth_ratio = detector.mouth_open_ratio(face, width, height)
```

**Methods:**
- `detect(frame)` → Returns frame, EAR value, face landmarks
- `head_tilt_angle(face, w, h)` → Returns head inclination angle
- `mouth_open_ratio(face, w, h)` → Returns vertical mouth distance

### `compute_ear()` Function (`src/detection/eye_aspect_ratio.py`)

Calculates Eye Aspect Ratio from 6 eye landmark points:

```python
from src.detection.eye_aspect_ratio import compute_ear

left_eye = [(x1,y1), (x2,y2), ..., (x6,y6)]
ear = compute_ear(left_eye)  # Returns float value
```

### `DriverSleepApp` Class (`src/gui/qt_app.py`)

PyQt5 application with continuous video processing:

```python
app = QApplication(sys.argv)
window = DriverSleepApp()
window.show()
sys.exit(app.exec_())
```

**Key features:**
- 30 FPS capture timer
- Real-time label updates
- Resolution switching
- Status color indication (red/green)

---

## ⚙️ Configuration & Tuning

### Adjust Sleep Detection Sensitivity

Edit thresholds in `src/app.py` or `src/gui/qt_app.py`:

```python
SLEEP_THRESHOLD = 0.60          # Lower = more sensitive
SLEEP_FRAMES_REQUIRED = 15      # Lower = faster alert
YAWN_THRESHOLD = 25             # Lower = detects smaller yawns
```

### Common Adjustments

| Parameter | Current | Action | Effect |
|-----------|---------|--------|--------|
| `SLEEP_THRESHOLD` | 0.60 | ↓ Decrease | More sensitive (fewer false negatives) |
| `SLEEP_THRESHOLD` | 0.60 | ↑ Increase | Less sensitive (fewer false positives) |
| `SLEEP_FRAMES_REQUIRED` | 15 | ↓ Decrease | Faster alerts |
| `SLEEP_FRAMES_REQUIRED` | 15 | ↑ Increase | More tolerant of brief eye closures |

---

## 🎓 Algorithm Flow Diagram

```
Video Frame
    ↓
MediaPipe FaceLandmarker Detection
    ↓
Extract Eye Landmarks
    ↓
Calculate EAR (Eye Aspect Ratio)
    ↓
Compare to SLEEP_THRESHOLD
    ├─ EAR ≥ Threshold → Eyes Open (counter = 0)
    └─ EAR < Threshold → Eyes Closed (counter++)
    ↓
Check: counter ≥ SLEEP_FRAMES_REQUIRED?
    ├─ NO → Continue
    └─ YES → ALERT! "SLEEP DETECTED"
    ↓
Also Check:
    ├─ Yawn (mouth distance > 25px)
    ├─ Head Tilt (angle > 0.25)
    └─ Blink Duration & Count
    ↓
Display Results on Frame
    ↓
Show via OpenCV or PyQt5
```

---

## 🔧 Troubleshooting

### Model Not Found Error
**Solution:** Ensure `face_landmarker.task` is in the `models/` folder

### Low FPS Performance
**Solution:** 
- Reduce resolution (640x480 recommended)
- Close other applications
- Check camera driver updates

### False Positives (Many Sleep Alerts)
**Solution:**
- Increase `SLEEP_THRESHOLD` value (e.g., 0.65 or 0.70)
- Increase `SLEEP_FRAMES_REQUIRED` (e.g., 20-25)

### Face Not Detected
**Solution:**
- Ensure adequate lighting
- Position face directly toward camera
- Check camera permissions
- Try `resolution_test.py` to verify camera works

---

## 📝 License & References

- **MediaPipe:** https://mediapipe.dev/
- **OpenCV:** https://opencv.org/
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
