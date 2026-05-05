"""MediaPipe Hands wrapper. Returns 21-point hand landmarks and draws them."""
import numpy as np
import mediapipe as mp
import cv2


class HandTracker:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.5, max_hands=1):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.hands.process(rgb)

    @staticmethod
    def landmarks_to_array(hand_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    def draw(self, image, results):
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return image

    def close(self):
        self.hands.close()
