"""Record landmark sequences for a sign label.

Usage:
    python collect_data.py hello
    python collect_data.py thanks 40         # record 40 sequences instead of default 30

Each sequence is 30 frames (~1 sec). A 3-second countdown precedes each capture
so you can get into position. Press 'q' at any time to stop early."""
import os
import sys
import time
import numpy as np
import cv2
from tracker import HolisticTracker

DATA_DIR = "data"
SEQUENCE_LENGTH = 30
DEFAULT_SEQUENCES = 30


def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_data.py <sign_label> [num_sequences]")
        sys.exit(1)

    sign = sys.argv[1].lower().strip()
    n_seq = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEQUENCES

    sign_dir = os.path.join(DATA_DIR, sign)
    os.makedirs(sign_dir, exist_ok=True)
    existing = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
    print(f"Sign '{sign}' already has {existing} sequence(s) on disk. Adding {n_seq} more.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera 0. If you have multiple cameras, edit collect_data.py to use a different index.")
        sys.exit(1)

    tracker = HolisticTracker()

    seq_idx = existing
    target_idx = existing + n_seq

    try:
        while seq_idx < target_idx:
            # Countdown
            countdown_start = time.time()
            while True:
                elapsed = time.time() - countdown_start
                remaining = 3 - int(elapsed)
                if remaining <= 0:
                    break
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                results = tracker.process(frame)
                tracker.draw(frame, results)
                cv2.putText(frame, f"Sign: {sign}   Seq {seq_idx + 1}/{target_idx}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Get ready: {remaining}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                cv2.imshow("Collecting", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    return

            # Record
            sequence = []
            while len(sequence) < SEQUENCE_LENGTH:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                results = tracker.process(frame)
                tracker.draw(frame, results)
                features = tracker.extract_features(results)
                sequence.append(features)
                cv2.putText(frame, f"REC {len(sequence)}/{SEQUENCE_LENGTH}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Sign: {sign}   Seq {seq_idx + 1}/{target_idx}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.imshow("Collecting", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    return

            arr = np.array(sequence, dtype=np.float32)
            out_path = os.path.join(sign_dir, f"seq_{seq_idx:04d}.npy")
            np.save(out_path, arr)
            print(f"  saved {out_path}")
            seq_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print(f"Done. Total sequences for '{sign}': {seq_idx}")


if __name__ == "__main__":
    main()
