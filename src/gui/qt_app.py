import sys
import math
import psutil
import time
import cv2
import numpy as np

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QFrame,
)

from src.camera.webcam import Webcam
from src.detection.face_mesh_detector import FaceMeshDetector

# Mobile simulation settings
MOBILE_SIMULATION = True
MOBILE_RESOLUTION = (320, 240)
SKIP_FRAMES = 2
ARTIFICIAL_DELAY = 0.02

SLEEP_THRESHOLD = 0.20
SLEEP_FRAMES_REQUIRED = 15
YAWN_THRESHOLD = 25


class DriverSleepApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Driver Sleep Detection - PyQt")
        self.resize(900, 700)

        # Core
        width, height = MOBILE_RESOLUTION if MOBILE_SIMULATION else (640, 480)
        self.cam = Webcam(width=width, height=height)
        self.detector = FaceMeshDetector()

        self.consecutive_sleep_frames = 0
        self.prev_time = time.time()

        self.blink_count = 0
        self.blink_start = None
        self.blink_durations = []

        # Mobile simulation state
        self.frame_idx = 0
        self.prev_ear = None
        self.prev_face = None

        # UI
        self._build_ui()

        # Timer de captura
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    def _build_ui(self):
        main_layout = QVBoxLayout()

        # Video
        self.video_label = QLabel()
        self.video_label.setFrameShape(QFrame.Box)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        main_layout.addWidget(self.video_label, stretch=3)

        # Stats
        stats_layout = QVBoxLayout()

        self.status_label = QLabel("Status: Awake")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        stats_layout.addWidget(self.status_label)

        self.ear_label = QLabel("EAR: -")
        self.fps_label = QLabel("FPS: -")
        self.blinks_label = QLabel("Blinks: 0")
        self.last_blink_label = QLabel("Last Blink: -")
        self.yawn_label = QLabel("Yawn: No")
        self.head_tilt_label = QLabel("Head Tilt: No")
        self.cpu_label = QLabel("CPU: -")
        self.hr_label = QLabel("HR: -")
        self.mobile_label = QLabel("Mobile Mode: Yes" if MOBILE_SIMULATION else "Mobile Mode: No")

        for lbl in [
            self.ear_label,
            self.fps_label,
            self.blinks_label,
            self.last_blink_label,
            self.yawn_label,
            self.head_tilt_label,
            self.cpu_label,
            self.hr_label,
            self.mobile_label,
        ]:
            lbl.setStyleSheet("font-size: 14px;")
            stats_layout.addWidget(lbl)

        # Controls
        controls_layout = QHBoxLayout()

        self.resolution_box = QComboBox()
        self.resolution_box.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution_box.currentIndexChanged.connect(self.change_resolution)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_detection)

        controls_layout.addWidget(QLabel("Resolution:"))
        controls_layout.addWidget(self.resolution_box)
        controls_layout.addStretch()
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        main_layout.addLayout(stats_layout, stretch=1)
        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)

    def change_resolution(self):
        text = self.resolution_box.currentText()
        w, h = map(int, text.split("x"))
        self.cam = Webcam(width=w, height=h)

    def start_detection(self):
        self.consecutive_sleep_frames = 0
        self.blink_count = 0
        self.blink_start = None
        self.blink_durations = []
        self.status_label.setText("Status: Detecting")

    def stop_detection(self):
        self.status_label.setText("Status: Stopped")

    def update_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            return

        self.frame_idx += 1

        # FPS
        now = time.time()
        fps = 1 / (now - self.prev_time)
        self.prev_time = now

        # Detección - Mobile simulation frame skipping
        if MOBILE_SIMULATION and self.frame_idx % SKIP_FRAMES != 0:
            ear = self.prev_ear
            face = self.prev_face
        else:
            frame, ear, face = self.detector.detect(frame)
            self.prev_ear = ear
            self.prev_face = face
            time.sleep(ARTIFICIAL_DELAY)

        h, w, _ = frame.shape

        head_tilt = False
        yawn = False

        if face is not None:
            angle = self.detector.head_tilt_angle(face, w, h)
            if angle > 0.25:
                head_tilt = True

            mouth_ratio = self.detector.mouth_open_ratio(face, w, h)
            if mouth_ratio > YAWN_THRESHOLD:
                yawn = True

        # EAR / parpadeos / sueño
        asleep = False

        if ear is not None:
            if ear < SLEEP_THRESHOLD:
                self.consecutive_sleep_frames += 1

                if self.blink_start is None:
                    self.blink_start = time.time()
            else:
                if self.blink_start is not None:
                    blink_duration = time.time() - self.blink_start
                    self.blink_durations.append(blink_duration)
                    self.blink_count += 1
                    self.blink_start = None

                self.consecutive_sleep_frames = 0

            if self.consecutive_sleep_frames >= SLEEP_FRAMES_REQUIRED:
                asleep = True

        # Actualizar labels
        self.ear_label.setText(f"EAR: {ear:.3f}" if ear is not None else "EAR: -")
        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.blinks_label.setText(f"Blinks: {self.blink_count}")
        if self.blink_durations:
            self.last_blink_label.setText(
                f"Last Blink: {self.blink_durations[-1]:.2f}s"
            )
        else:
            self.last_blink_label.setText("Last Blink: -")

        self.yawn_label.setText(f"Yawn: {'Yes' if yawn else 'No'}")
        self.head_tilt_label.setText(f"Head Tilt: {'Yes' if head_tilt else 'No'}")

        # Mobile simulation stats
        cpu_usage = psutil.cpu_percent(interval=None)
        heart_rate = 70 + int(5 * math.sin(time.time() / 3.0))
        self.cpu_label.setText(f"CPU: {cpu_usage:.0f}%")
        self.hr_label.setText(f"HR: {heart_rate} bpm")
        self.mobile_label.setText("Mobile Mode: Yes" if MOBILE_SIMULATION else "Mobile Mode: No")

        # Status + color
        if asleep:
            self.status_label.setText("Status: SLEEP DETECTED")
            self.status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: red;"
            )
        else:
            self.status_label.setText("Status: Detecting")
            self.status_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: green;"
            )

        # Mostrar frame en QLabel (reescala para UI, no para procesamiento)
        frame = cv2.resize(frame, (800, 450), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        self.cam.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = DriverSleepApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()