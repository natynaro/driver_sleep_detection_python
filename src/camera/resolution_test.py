import cv2
import time

def test_resolution(width, height):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        cv2.putText(frame, f"{width}x{height} | FPS: {fps:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Resolution Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
