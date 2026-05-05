"""Live sign-language to speech. No training, no recording - just run it.

Recognizes ASL fingerspelling letters (A B D F I L U V W Y) and common signs
(yes / hello / I love you). Letters get appended to the current word. The word
is spoken when your hand leaves view, when you press SPACE, or after you've
held a different gesture.

Keys:
  q          quit
  c          clear current word
  space      speak the current word now
  s          repeat the last spoken phrase
"""
import time
import threading
import cv2
import pyttsx3

from tracker import HandTracker
from recognizer import classify


STABLE_FRAMES = 8        # frames a label must hold to be accepted
NEW_LETTER_GAP = 12      # frames between accepting the same letter twice in a row
WORD_TIMEOUT = 1.2       # seconds with no hand visible -> speak the word


class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)
        self._lock = threading.Lock()

    def say(self, text):
        if not text:
            return
        threading.Thread(target=self._say_blocking, args=(text,), daemon=True).start()

    def _say_blocking(self, text):
        with self._lock:
            self.engine.say(text)
            self.engine.runAndWait()


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera. Make sure no other app is using it.")
        return

    tracker = HandTracker()
    speaker = Speaker()
    speaker.say("Sign translator ready")
    print("Sign Translator running. Press q to quit.")

    current_word = ""
    last_label = ""
    label_streak = 0
    frames_since_committed = 999
    last_hand_seen = time.time()
    history = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            results = tracker.process(frame)

            label, conf = "?", 0.0
            now = time.time()

            if results.multi_hand_landmarks:
                last_hand_seen = now
                pts = HandTracker.landmarks_to_array(results.multi_hand_landmarks[0])
                label, conf = classify(pts)
                tracker.draw(frame, results)

                if label != "?" and label == last_label:
                    label_streak += 1
                else:
                    last_label = label
                    label_streak = 1

                if label_streak == STABLE_FRAMES and label != "?":
                    if len(label) == 1:
                        # single letter
                        if not current_word or current_word[-1] != label or frames_since_committed > NEW_LETTER_GAP:
                            current_word += label
                            frames_since_committed = 0
                    else:
                        # full word/phrase: flush any pending letters first
                        if current_word:
                            speaker.say(current_word)
                            history.append(current_word)
                            current_word = ""
                        speaker.say(label)
                        history.append(label)
                        frames_since_committed = 0

                frames_since_committed += 1
            else:
                if current_word and (now - last_hand_seen) > WORD_TIMEOUT:
                    speaker.say(current_word)
                    history.append(current_word)
                    current_word = ""
                    last_label = ""
                    label_streak = 0

            # HUD
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Word: {current_word}_", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            recent = " ".join(history[-5:]) if history else "(spoken history)"
            cv2.putText(frame, recent, (10, 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
            if label != "?":
                bar = int(min(conf, 1.0) * 200)
                cv2.rectangle(frame, (10, h - 25), (10 + bar, h - 13), (0, 255, 0), -1)
                stab = min(label_streak, STABLE_FRAMES)
                cv2.putText(frame, f"{label}  conf {conf:.2f}  hold {stab}/{STABLE_FRAMES}",
                            (220, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.putText(frame, "q=quit  c=clear  space=speak",
                        (w - 270, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Sign Translator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_word = ""
                last_label = ""
                label_streak = 0
            elif key == ord(' ') and current_word:
                speaker.say(current_word)
                history.append(current_word)
                current_word = ""
            elif key == ord('s') and history:
                speaker.say(history[-1])
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
