import cv2
import mediapipe as mp
import numpy as np
from typing import Any, Sequence

class BlinkDetectionModel:
    """
    Modular blink detection model using MediaPipe Face Mesh and EAR algorithm.
    Designed for both real-time and batch evaluation.
    """
    ear_threshold: float
    consecutive_frames: int
    min_detection_confidence: float
    min_tracking_confidence: float
    left_eye_indices: list[int]
    right_eye_indices: list[int]
    mp_face_mesh: Any
    face_mesh: Any

    def __init__(self, 
                 ear_threshold: float = 0.21,
                 consecutive_frames: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        self.mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
        self.face_mesh = None

    def _euclidean_dist(self, pt1: tuple[float, float], pt2: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))
    
    def _eye_aspect_ratio(self, eye_landmarks: Sequence[tuple[float, float]]) -> float:
        A = self._euclidean_dist(eye_landmarks[1], eye_landmarks[5])
        B = self._euclidean_dist(eye_landmarks[2], eye_landmarks[4])
        C = self._euclidean_dist(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _initialize_face_mesh(self) -> None:
        if self.face_mesh is None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(  # type: ignore[attr-defined]
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
    
    def process_frame(self, frame: np.ndarray[Any, Any]) -> dict[str, Any]:
        self._initialize_face_mesh()
        assert self.face_mesh is not None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)  # type: ignore[attr-defined]
        if not getattr(results, 'multi_face_landmarks', None):
            return {
                'ear': None,
                'blink_detected': False,
                'face_detected': False,
                'left_ear': None,
                'right_ear': None
            }
        h, w, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]
        left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in self.left_eye_indices]
        right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in self.right_eye_indices]
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        return {
            'ear': ear,
            'blink_detected': ear < self.ear_threshold,
            'face_detected': True,
            'left_ear': left_ear,
            'right_ear': right_ear
        }
    
    def process_video(self, video_path: str, output_frames: bool = False) -> dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        total_frames = 0
        frames_with_face = 0
        frames_with_blink = 0
        ear_values: list[float] = []
        blink_frames: list[int] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.process_frame(frame)
                total_frames += 1
                if result['face_detected']:
                    frames_with_face += 1
                    ear_values.append(result['ear'])
                    if result['blink_detected']:
                        frames_with_blink += 1
                        blink_frames.append(total_frames - 1)
        finally:
            cap.release()
        average_ear = float(np.mean(ear_values)) if ear_values else 0.0
        return {
            'total_frames': total_frames,
            'frames_with_face': frames_with_face,
            'frames_with_blink': frames_with_blink,
            'ear_values': ear_values if output_frames else None,
            'blink_frames': blink_frames,
            'average_ear': average_ear,
            'face_detection_rate': frames_with_face / total_frames if total_frames > 0 else 0.0,
            'blink_rate': frames_with_blink / frames_with_face if frames_with_face > 0 else 0.0
        }
    
    def cleanup(self) -> None:
        if self.face_mesh:
            self.face_mesh.close()  # type: ignore[attr-defined]
            self.face_mesh = None 