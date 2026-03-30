import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .eye_aspect_ratio import compute_ear


class FaceWrapper:
    """
    Pequeño wrapper para imitar la interfaz antigua:
    face.landmark[idx].x / .y
    """
    def __init__(self, landmarks):
        self.landmark = landmarks


class FaceMeshDetector:
    def __init__(self, model_path="models/face_landmarker.task"):
        # Carga del modelo FaceLandmarker (MediaPipe 0.10.x)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Índices de los ojos (MediaPipe FaceMesh / FaceLandmarker)
        self.left_eye_idx = [159, 145, 153, 154, 155, 133]
        self.right_eye_idx = [386, 374, 380, 381, 382, 362]

    def detect(self, frame):
        """
        Devuelve:
        - frame (sin modificar)
        - avg_ear (float o None)
        - face (FaceWrapper o None)
        """
        h, w, _ = frame.shape

        # Convertir a RGB y a mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detección
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return frame, None, None

        landmarks = result.face_landmarks[0]  # lista de NormalizedLandmark
        face = FaceWrapper(landmarks)

        left_eye = []
        right_eye = []

        for idx in self.left_eye_idx:
            lm = landmarks[idx]
            left_eye.append((lm.x * w, lm.y * h))

        for idx in self.right_eye_idx:
            lm = landmarks[idx]
            right_eye.append((lm.x * w, lm.y * h))

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2

        return frame, avg_ear, face

    def head_tilt_angle(self, face, w, h):
        """
        Inclinación de cabeza usando nariz (1) y mentón (152).
        """
        nose = face.landmark[1]
        chin = face.landmark[152]

        x1, y1 = nose.x * w, nose.y * h
        x2, y2 = chin.x * w, chin.y * h

        angle = abs((x2 - x1) / (y2 - y1 + 1e-6))
        return angle

    def mouth_open_ratio(self, face, w, h):
        """
        Bostezo: distancia vertical entre labio superior (13) e inferior (14).
        """
        top = face.landmark[13]
        bottom = face.landmark[14]

        top_y = top.y * h
        bottom_y = bottom.y * h

        return abs(bottom_y - top_y)
