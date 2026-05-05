import cv2
import math
import time
import psutil
from camera.webcam import Webcam
from detection.face_mesh_detector import FaceMeshDetector

MOBILE_SIMULATION = True
MOBILE_RESOLUTION = (320, 240)
SKIP_FRAMES = 2
ARTIFICIAL_DELAY = 0.02

SLEEP_THRESHOLD = 0.60
SLEEP_FRAMES_REQUIRED = 15  # ~0.5 seg 30 FPS
YAWN_THRESHOLD = 25         # vertical distance between each lip

def main():
    width, height = MOBILE_RESOLUTION if MOBILE_SIMULATION else (640, 480)
    cam = Webcam(width=width, height=height)
    detector = FaceMeshDetector()

    consecutive_sleep_frames = 0

    # FPS
    prev_time = time.time()

    # blinks
    blink_count = 0
    blink_start = None
    blink_durations = []

    # Mobile simulation state
    frame_idx = 0
    prev_ear = None
    prev_face = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_idx += 1

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        # Detection FaceMesh
        if MOBILE_SIMULATION and frame_idx % SKIP_FRAMES != 0:
            ear = prev_ear
            face = prev_face
        else:
            frame, ear, face = detector.detect(frame)
            prev_ear = ear
            prev_face = face
            time.sleep(ARTIFICIAL_DELAY)

        if face is not None:
            h, w, _ = frame.shape

            # -------------------------
            # 1. Head inclination
            # -------------------------
            angle = detector.head_tilt_angle(face, w, h)
            if angle > 0.25:
                cv2.putText(frame, "Head Tilt Detected", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # -------------------------
            # 2. Yawn detection
            # -------------------------
            mouth_ratio = detector.mouth_open_ratio(face, w, h)
            if mouth_ratio > YAWN_THRESHOLD:
                cv2.putText(frame, "Yawn Detected", (10, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

        # -------------------------
        # 3. EAR (Eye Aspect Ratio)
        # -------------------------
        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Closed eyes
            if ear < SLEEP_THRESHOLD:
                consecutive_sleep_frames += 1
                cv2.putText(frame, "Eyes Closed", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # Blink count initiation
                if blink_start is None:
                    blink_start = time.time()

            else:
                # If blinking, end blink
                if blink_start is not None:
                    blink_duration = time.time() - blink_start
                    blink_durations.append(blink_duration)
                    blink_count += 1
                    blink_start = None

                consecutive_sleep_frames = 0

            # Sleep detection alert
            if consecutive_sleep_frames >= SLEEP_FRAMES_REQUIRED:
                cv2.putText(frame, "SLEEP DETECTED!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # -------------------------
        # 4. Show statistics
        # -------------------------
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.putText(frame, f"Blinks: {blink_count}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        if blink_durations:
            cv2.putText(frame, f"Last Blink: {blink_durations[-1]:.2f}s", (10, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cpu_usage = psutil.cpu_percent(interval=None)
        heart_rate = 70 + int(5 * math.sin(time.time() / 3.0))
        cv2.putText(frame, f"CPU: {cpu_usage:.0f}%", (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"HR: {heart_rate} bpm", (10, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Mobile Mode: {'Yes' if MOBILE_SIMULATION else 'No'}", (10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # -------------------------
        # Show window
        # -------------------------
        cv2.imshow("Driver Sleep Detection (Python)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cam.release()

if __name__ == "__main__":
    main()