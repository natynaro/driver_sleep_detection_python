import cv2
import time
from camera.webcam import Webcam
from detection.face_mesh_detector import FaceMeshDetector

SLEEP_THRESHOLD = 0.60
SLEEP_FRAMES_REQUIRED = 15  # ~0.5 segundos a 30 FPS
YAWN_THRESHOLD = 25         # distancia vertical entre labios

def main():
    cam = Webcam(width=640, height=480)
    detector = FaceMeshDetector()

    consecutive_sleep_frames = 0

    # FPS
    prev_time = time.time()

    # Parpadeos
    blink_count = 0
    blink_start = None
    blink_durations = []

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        # Detección FaceMesh
        frame, ear, face = detector.detect(frame)

        if face is not None:
            h, w, _ = frame.shape

            # -------------------------
            # 1. Inclinación de cabeza
            # -------------------------
            angle = detector.head_tilt_angle(face, w, h)
            if angle > 0.25:
                cv2.putText(frame, "Head Tilt Detected", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # -------------------------
            # 2. Detección de bostezo
            # -------------------------
            mouth_ratio = detector.mouth_open_ratio(face, w, h)
            if mouth_ratio > YAWN_THRESHOLD:
                cv2.putText(frame, "Yawn Detected", (10, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

        # -------------------------
        # 3. EAR (ojos)
        # -------------------------
        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Ojos cerrados
            if ear < SLEEP_THRESHOLD:
                consecutive_sleep_frames += 1
                cv2.putText(frame, "Eyes Closed", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # Inicia conteo de parpadeo
                if blink_start is None:
                    blink_start = time.time()

            else:
                # Si estaba parpadeando, termina el parpadeo
                if blink_start is not None:
                    blink_duration = time.time() - blink_start
                    blink_durations.append(blink_duration)
                    blink_count += 1
                    blink_start = None

                consecutive_sleep_frames = 0

            # Detección de sueño
            if consecutive_sleep_frames >= SLEEP_FRAMES_REQUIRED:
                cv2.putText(frame, "SLEEP DETECTED!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # -------------------------
        # 4. Mostrar estadísticas
        # -------------------------
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.putText(frame, f"Blinks: {blink_count}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        if blink_durations:
            cv2.putText(frame, f"Last Blink: {blink_durations[-1]:.2f}s", (10, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        # -------------------------
        # Mostrar ventana
        # -------------------------
        cv2.imshow("Driver Sleep Detection (Python)", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cam.release()

if __name__ == "__main__":
    main()