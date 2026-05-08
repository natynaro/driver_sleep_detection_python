import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import time
import psutil
import cv2
import pandas as pd
import numpy as np

from enum import Enum
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QFrame
)

from src.camera.webcam import Webcam
from src.detection.face_mesh_detector import FaceMeshDetector
from src.metrics.metrics_collector import MetricsCollector, FusedDriverState


# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================

# Camera & processing
MOBILE_RESOLUTION = (640, 480)
FPS = 30.0
DT = 1.0 / FPS

# Eye aspect ratio (EAR)
EYE_OPEN_THRESHOLD = 0.75
EYE_CLOSED_RATIO = 0.92
MICROSLEEP_FRAMES = 8          # ~0.25s

# Facial cues
YAWN_RATIO_THRESHOLD = 0.12
HEAD_TILT_THRESHOLD = 0.15

# Temporal logic (seconds)
YAWN_TILT_WINDOW = 2.0
HEAD_TILT_SUSTAINED = 2.0
AWAKE_RESET_TIME = 4.0


# ============================================================
# DRIVER STATE DEFINITION
# ============================================================

class DriverState(Enum):
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    SLEEPY = "SLEEPY"


# ============================================================
# MAIN APPLICATION
# ============================================================

class DriverSleepApp(QWidget):
    """
    Visual-only driver drowsiness detection
    using physiologically meaningful rules.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Driver Sleep Detection – Visual Model")
        self.resize(950, 750)

        # --------------------------------------------------------
        # Camera and detector
        # --------------------------------------------------------
        self.cam = Webcam(width=MOBILE_RESOLUTION[0], height=MOBILE_RESOLUTION[1])
        self.detector = FaceMeshDetector()

        # --------------------------------------------------------
        # Wearable data loading and processing
        # --------------------------------------------------------
        self.wearable_data = pd.read_excel('src/data/wearable_ble_stream.xlsx')
        self.wearable_data['t_arrival_s'] = self.wearable_data['t_arrival_ms'] / 1000.0
        self.wearable_times = self.wearable_data['t_arrival_s'].values
        self.wearable_levels = self.wearable_data['drowsiness_level'].values
        self.last_wearable_level = None  # For packet loss handling

        # --------------------------------------------------------
        # Metrics collector
        # --------------------------------------------------------
        self.metrics = MetricsCollector()

        # --------------------------------------------------------
        # Session timing
        # --------------------------------------------------------
        self.start_time = time.time()

        # --------------------------------------------------------
        # EAR (eye aspect ratio) tracking
        # --------------------------------------------------------
        self.ear_baseline = None
        self.eye_closed_frames = 0

        # --------------------------------------------------------
        # Temporal markers (seconds)
        # --------------------------------------------------------
        self.time_since_yawn = 999.0
        self.time_since_tilt = 999.0
        self.time_since_normal = 0.0

        # --------------------------------------------------------
        # Blink monitoring (fatigue indicator)
        # --------------------------------------------------------
        self.recent_blinks = 0
        self.blink_timer = 0.0

        # --------------------------------------------------------
        # Current driver state
        # --------------------------------------------------------
        self.driver_state = DriverState.AWAKE
        self.fused_state = FusedDriverState.AWAKE

        # --------------------------------------------------------
        # User interface
        # --------------------------------------------------------
        self._build_ui()

        # --------------------------------------------------------
        # Timer (main loop)
        # --------------------------------------------------------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / FPS))

    # =========================================================
    # UI CREATION
    # =========================================================

    def _build_ui(self):
        layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setFrameShape(QFrame.Box)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        self.state_label = QLabel("Visual Model: AWAKE")
        self.state_label.setStyleSheet("font-size:18px;font-weight:bold")
        layout.addWidget(self.state_label)

        self.ear_label = QLabel("EAR: -")
        self.yawn_label = QLabel("Yawn: No")
        self.tilt_label = QLabel("Head Tilt: No")
        self.microsleep_label = QLabel("Microsleep: No")

        # New labels for evaluation
        self.fused_state_label = QLabel("Fused Model: AWAKE")
        self.fused_state_label.setStyleSheet("font-size:18px;font-weight:bold;color:blue")
        self.wearable_label = QLabel("Wearable: AWAKE")
        self.metrics_label = QLabel("CPU: -% | Delay: -s | Coinc: -%")
        self.timeline_label = QLabel("Timeline: -")

        # Alert button that simulates an in-window driver warning
        self.alert_button = QPushButton("DRIVER STATUS: NORMAL")
        self.alert_button.setEnabled(False)
        self.alert_button.setStyleSheet(
            "font-size:16px; font-weight:bold; background-color: green; color: white;"
        )

        layout.addWidget(self.ear_label)
        layout.addWidget(self.yawn_label)
        layout.addWidget(self.tilt_label)
        layout.addWidget(self.microsleep_label)
        layout.addWidget(self.fused_state_label)
        layout.addWidget(self.wearable_label)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.timeline_label)
        layout.addWidget(self.alert_button)

        self.setLayout(layout)

    # =========================================================
    # WEARABLE DATA PROCESSING METHODS
    # =========================================================

    def _get_wearable_level(self, t):
        """
        Interpolate wearable drowsiness level at time t.

        Args:
            t (float): Current time in seconds.

        Returns:
            float or None: Interpolated drowsiness level, or None if no data.
        """
        if len(self.wearable_times) == 0:
            return None

        # Handle packet loss: if packet_lost=1, use last valid value
        valid_indices = self.wearable_data['packet_lost'] == 0
        if not valid_indices.any():
            return self.last_wearable_level

        valid_times = self.wearable_times[valid_indices]
        valid_levels = self.wearable_levels[valid_indices]

        if t < valid_times[0]:
            return valid_levels[0]
        elif t > valid_times[-1]:
            return valid_levels[-1]
        else:
            level = np.interp(t, valid_times, valid_levels)
            self.last_wearable_level = level
            return level

    def _fuse_states(self, visual_state, wearable_drowsy):
        """
        Fuse visual and wearable states using conservative logic.

        Args:
            visual_state (DriverState): Current visual state.
            wearable_drowsy (bool): True if wearable indicates DROWSY.

        Returns:
            FusedDriverState: Fused state.
        """
        if visual_state == DriverState.SLEEPY:
            return FusedDriverState.SLEEPY
        elif visual_state == DriverState.DROWSY or wearable_drowsy:
            return FusedDriverState.DROWSY
        else:
            return FusedDriverState.AWAKE

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def update_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            return

        frame, ear, face = self.detector.detect(frame)

        # -----------------------------------------------------
        # Initialize variables
        # -----------------------------------------------------
        yawn = False
        head_tilt = False
        microsleep = False

        # -----------------------------------------------------
        # EAR processing and microsleep detection
        # -----------------------------------------------------
        if ear is not None:
            if self.ear_baseline is None:
                self.ear_baseline = ear

            if ear < self.ear_baseline * EYE_CLOSED_RATIO:
                self.eye_closed_frames += 1
            else:
                self.eye_closed_frames = 0

            if ear > self.ear_baseline * 0.98:
                self.ear_baseline = (
                    0.95 * self.ear_baseline + 0.05 * ear
                )

        microsleep = self.eye_closed_frames >= MICROSLEEP_FRAMES

        # -----------------------------------------------------
        # Wearable data processing
        # -----------------------------------------------------
        current_time = time.time() - self.start_time
        wearable_level = self._get_wearable_level(current_time)
        wearable_drowsy = wearable_level > 0.6 if wearable_level is not None else False

        # -----------------------------------------------------
        # STATE MACHINE (REALISTIC LOGIC)
        # -----------------------------------------------------
        if face is not None:
            mouth_open = self.detector.mouth_open_ratio(
                face, frame.shape[1], frame.shape[0]
            )
            face_height = abs(
                face.landmark[152].y - face.landmark[10].y
            ) * frame.shape[0]

            if face_height > 0 and (mouth_open / face_height) > YAWN_RATIO_THRESHOLD:
                yawn = True
                self.time_since_yawn = 0.0
            else:
                self.time_since_yawn += DT

            tilt_angle = self.detector.head_tilt_angle(
                face, frame.shape[1], frame.shape[0]
            )

            if tilt_angle > HEAD_TILT_THRESHOLD:
                head_tilt = True
                self.time_since_tilt += DT
            else:
                self.time_since_tilt = 0.0
        else:
            self.time_since_yawn += DT
            self.time_since_tilt += DT

        # -----------------------------------------------------
        # STATE MACHINE (REALISTIC LOGIC)
        # -----------------------------------------------------

        # Rule 1: Microsleep → SLEEPY
        if microsleep:
            self.driver_state = DriverState.SLEEPY
            self.time_since_normal = 0.0

        # Rule 2: Yawn + head tilt close in time → DROWSY
        elif self.time_since_yawn < YAWN_TILT_WINDOW and self.time_since_tilt < YAWN_TILT_WINDOW:
            self.driver_state = DriverState.DROWSY
            self.time_since_normal = 0.0

        # Rule 3: Sustained head tilt → DROWSY
        elif self.time_since_tilt > HEAD_TILT_SUSTAINED:
            self.driver_state = DriverState.DROWSY
            self.time_since_normal = 0.0

        # Rule 4: Return to AWAKE after stable normal behavior
        else:
            self.time_since_normal += DT
            if self.time_since_normal > AWAKE_RESET_TIME:
                self.driver_state = DriverState.AWAKE

        # -----------------------------------------------------
        # State fusion and metrics update
        # -----------------------------------------------------
        self.fused_state = self._fuse_states(self.driver_state, wearable_drowsy)
        self.metrics.update(current_time, self.driver_state, wearable_drowsy, self.fused_state, psutil.cpu_percent())

        # -----------------------------------------------------
        # UI update
        # -----------------------------------------------------

        self.state_label.setText(f"Visual Model: {self.driver_state.value}")
        self.ear_label.setText(f"EAR: {ear:.3f}" if ear else "EAR: -")
        self.yawn_label.setText(f"Yawn: {'Yes' if yawn else 'No'}")
        self.tilt_label.setText(f"Head Tilt: {'Yes' if head_tilt else 'No'}")
        self.microsleep_label.setText(
            f"Microsleep: {'Yes' if microsleep else 'No'}"
        )

        # Update new labels
        self.fused_state_label.setText(f"Fused Model: {self.fused_state.value}")
        if wearable_level is None:
            self.wearable_label.setText("Wearable: No data")
        else:
            wearable_state = "DROWSY" if wearable_drowsy else "AWAKE"
            self.wearable_label.setText(f"Wearable: {wearable_state} ({wearable_level:.2f})")

        cpu_avg = self.metrics.get_average_cpu()
        delay = self.metrics.get_detection_delay('fused', current_time=current_time)
        coinc = self.metrics.get_coincidence_percentage()
        delay_text = f"{delay:.2f}s" if delay is not None else "-"
        self.metrics_label.setText(
            f"CPU avg: {cpu_avg:.1f}% | Delay: {delay_text} | Coinc: {coinc:.1f}%"
        )
        timeline = " | ".join(self.metrics.get_temporal_events(5))
        self.timeline_label.setText(f"Timeline: {timeline}")

        # Update alert button status in the same window
        if self.fused_state == FusedDriverState.SLEEPY:
            self.alert_button.setText("WARNING: MICROSLEEP DETECTED")
            self.alert_button.setStyleSheet(
                "font-size:16px; font-weight:bold; background-color: red; color: white;"
            )
        elif self.fused_state == FusedDriverState.DROWSY:
            self.alert_button.setText("WARNING: DROWSY DRIVER")
            self.alert_button.setStyleSheet(
                "font-size:16px; font-weight:bold; background-color: orange; color: black;"
            )
        else:
            self.alert_button.setText("DRIVER STATUS: NORMAL")
            self.alert_button.setStyleSheet(
                "font-size:16px; font-weight:bold; background-color: green; color: white;"
            )

        frame = cv2.resize(frame, (800, 450))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.video_label.setPixmap(QPixmap.fromImage(
            QImage(
                rgb.data, rgb.shape[1], rgb.shape[0],
                rgb.strides[0], QImage.Format_RGB888
            )
        ))

    def closeEvent(self, event):
        self.cam.release()
        # Export metrics for analysis
        self.metrics.export_to_csv('data/session_logs.csv')
        event.accept()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    app = QApplication(sys.argv)
    window = DriverSleepApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()