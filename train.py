"""Train an LSTM classifier on the recorded sequences in data/.

Each subdirectory of data/ is one sign label. Each .npy file is a 30 x 1662
landmark sequence."""
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Quiet TF logging.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data"
MODELS_DIR = "models"
SEQUENCE_LENGTH = 30


def load_data():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"No '{DATA_DIR}/' folder. Run collect_data.py first.")
    signs = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
    X, y = [], []
    for label_idx, sign in enumerate(signs):
        sign_dir = os.path.join(DATA_DIR, sign)
        for fname in sorted(os.listdir(sign_dir)):
            if not fname.endswith('.npy'):
                continue
            seq = np.load(os.path.join(sign_dir, fname))
            if seq.shape[0] == SEQUENCE_LENGTH:
                X.append(seq)
                y.append(label_idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), signs


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y, signs = load_data()
    print(f"Loaded {len(X)} sequences across {len(signs)} sign(s): {signs}")
    if len(signs) < 2:
        print("Need at least 2 different signs to train. Record more with collect_data.py.")
        return
    if len(X) < 4 * len(signs):
        print("Warning: very small dataset. Aim for >= 30 sequences per sign for decent accuracy.")

    y_cat = to_categorical(y, num_classes=len(signs))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42,
        stratify=y if min(np.bincount(y)) >= 2 else None,
    )

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, X.shape[2])),
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(len(signs), activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss', min_lr=1e-5),
    ]
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test),
              callbacks=callbacks, batch_size=16, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.3f}")

    model_path = os.path.join(MODELS_DIR, "sign_lstm.keras")
    model.save(model_path)
    with open(os.path.join(MODELS_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        for s in signs:
            f.write(s + "\n")
    print(f"Saved model to {model_path}")
    print(f"Saved {len(signs)} labels to {os.path.join(MODELS_DIR, 'labels.txt')}")


if __name__ == "__main__":
    main()
