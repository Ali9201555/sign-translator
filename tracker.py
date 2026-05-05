"""MediaPipe Holistic wrapper. Extracts pose + face + both-hands landmarks per frame
and flattens them into a fixed-length feature vector for the model."""
import numpy as np
import mediapipe as mp
import cv2

# 33 pose landmarks * (x,y,z,visibility) = 132
# 468 face landmarks * (x,y,z)            = 1404
# 21 left-hand landmarks * (x,y,z)        = 63
# 21 right-hand landmarks * (x,y,z)       = 63
# total per frame                         = 1662
FEATURE_DIM = 33 * 4 + 468 * 3 + 21 * 3 + 21 * 3


class HolisticTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        return results

    def extract_features(self, results):
        pose = (
            np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4)
        )
        face = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(468 * 3)
        )
        lh = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3)
        )
        rh = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3)
        )
        return np.concatenate([pose, face, lh, rh])

    def has_hands(self, results):
        return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

    def draw(self, image, results):
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
            )
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )
        return image

    def close(self):
        self.holistic.close()
