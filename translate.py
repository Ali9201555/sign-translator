"""Live sign-language to speech.

- Captures webcam frames continuously
- Maintains a 30-frame sliding window of landmark vectors
- Runs LSTM inference on the window every frame
- When a sign is held with confidence > threshold for several frames in a row,
  it's appended to the sentence and spoken via SAPI5 (pyttsx3)

Keys:
  q   quit
  c   clear current sentence
  s   speak the current sentence again"""
import os
import threading
from collections import deque
import numpy as np
import cv2
import pyttsx3

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow.keras.models import load_model
from tracker import HolisticTracker

MODEL_PATH = "models/sign_lstm.keras"
LABELS_PATH = "models/labels.txt"
SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.85
STABLE_FRAMES = 5  # how many consecutive frames a prediction must hold


class Speaker:
    """Background TTS so speech doesn't block the camera loop."""
    def __init__(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', 175)
        self._lock = threading.Lock()

    def say(self, text):
        threading.Thread(target=self._say_blocking, args=(text,), daemon=True).start()

    def _say_blocking(self, text):
        with self._lock:
            self._engine.say(text)
            self._engine.runAndWait()


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH}.")
        print("Run collect_data.py for at least 2 signs, then train.py first.")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(labels)} sign(s): {labels}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    tracker = HolisticTracker()
    speaker = Speaker()

    window = deque(maxlen=SEQUENCE_LENGTH)
    sentence = []
    last_pred = None
    consecutive = 0
    last_conf = 0.0
    last_label = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            results = tracker.process(frame)
            tracker.draw(frame, results)
            features = tracker.extract_features(results)
            window.append(features)

            if len(window) == SEQUENCE_LENGTH and tracker.has_hands(results):
                X = np.expand_dims(np.array(window, dtype=np.float32), axis=0)
                probs = model.predict(X, verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                last_label = labels[idx]
                last_conf = conf

                if conf > CONF_THRESHOLD:
                    if last_label == last_pred:
                        consecutive += 1
                    else:
                        last_pred = last_label
                        consecutive = 1

                    if consecutive == STABLE_FRAMES:
                        if not sentence or sentence[-1] != last_label:
                            sentence.append(last_label)
                            speaker.say(last_label)
                            if len(sentence) > 10:
                                sentence = sentence[-10:]
                else:
                    consecutive = 0
            else:
                last_label = ""
                last_conf = 0.0
                consecutive = 0

            # HUD
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 0), -1)
            cv2.putText(frame, " ".join(sentence) if sentence else "(speak with your hands...)",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
            if last_label:
                bar = int(min(last_conf, 1.0) * 200)
                cv2.rectangle(frame, (10, h - 22), (10 + bar, h - 12), (0, 255, 0), -1)
                cv2.putText(frame, f"{last_label}  {last_conf:.2f}",
                            (220, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "q=quit  c=clear  s=replay",
                        (w - 240, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Sign Translator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []
                last_pred = None
                consecutive = 0
            elif key == ord('s') and sentence:
                speaker.say(" ".join(sentence))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
