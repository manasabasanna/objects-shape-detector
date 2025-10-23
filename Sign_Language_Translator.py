"""
Sign_Language_Translator.py

Single-file toolkit with three modes:
 1) collect  - collect landmark data from webcam for labels you choose
 2) train    - train a classifier on collected data and save model
 3) translate- real-time translator using the trained model

Dependencies: mediapipe, opencv-python, scikit-learn, pandas, numpy, joblib
Install: pip install mediapipe opencv-python scikit-learn pandas numpy joblib

Usage examples:
  python Sign_Language_Translator.py collect --label A --samples 200
  python Sign_Language_Translator.py train --data collected_data.csv --model model.joblib
  python Sign_Language_Translator.py translate --model model.joblib

Notes:
 - This is a practical, easy-to-run pipeline intended for ASL letters/words but requires you to collect sample data.
 - Collect clear samples from a consistent background and camera angle for best results.

"""

import argparse
import csv
import os
import time
from collections import deque

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe is required. Install with: pip install mediapipe")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(results):
    """Return flattened landmark list for both hands (42 coords each -> 84 values) or Nones filled if missing."""
    # We'll create a fixed-length feature: 21 points * 3 coords * 2 hands = 126 values
    # Order: right hand (x,y,z) for 21 landmarks, then left hand. If a hand missing -> zeros.
    features = []
    # Mediapipe doesn't declare left/right consistently depending on orientation; we'll use multi_hand_landmarks order.
    if not results.multi_hand_landmarks:
        return [0.0] * 126

    # Initialize with zeros
    hands = [[0.0] * 63, [0.0] * 63]
    # Fill in up to two hands (first two)
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
        vals = []
        for lm in hand_landmarks.landmark:
            vals.extend([lm.x, lm.y, lm.z])
        hands[i] = vals
    features = hands[0] + hands[1]
    return features


def collect_data(output_csv, label, samples=300, camera_id=0):
    """Collect landmark data and write to CSV. Press 'q' to abort early."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Could not open camera")
        return

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        collected = 0
        fields = [f'f{i}' for i in range(126)] + ['label']
        # create CSV if not exists
        file_exists = os.path.isfile(output_csv)
        csvfile = open(output_csv, 'a', newline='')
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(fields)

        print(f"Collecting {samples} samples for label '{label}'. Press space to record a sample. 'q' to quit.")
        print("Tip: Keep consistent pose and background. Aim for different rotations/lighting for robustness.")

        while collected < samples:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[0]) if results.multi_hand_landmarks else None

            cv2.putText(image, f"Collected: {collected}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f"Label: {label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Collect - Press SPACE to capture', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:  # SPACE pressed -> capture
                feats = extract_landmarks(results)
                writer.writerow(feats + [label])
                collected += 1
                print(f"Captured sample {collected}")

        csvfile.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Data collection finished.")


def train_model(data_csv, model_out, test_size=0.2, random_state=42):
    df = pd.read_csv(data_csv)
    if 'label' not in df.columns:
        raise ValueError('CSV must include a label column named "label"')
    X = df.drop('label', axis=1).values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    print("Training classifier... This may take a minute depending on data size.")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Train finished. Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")


def real_time_translate(model_path, camera_id=0, smoothing=5):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Train a model first.")
    clf = joblib.load(model_path)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Could not open camera")
        return

    label_history = deque(maxlen=smoothing)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        print("Starting real-time translation. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

            feats = np.array(extract_landmarks(results)).reshape(1, -1)
            pred = clf.predict(feats)[0]
            # smoothing: take the most common label in the recent history
            label_history.append(pred)
            display_label = max(set(label_history), key=label_history.count)

            # overlay the predicted label
            cv2.putText(image, f"Predicted: {display_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow('Sign Language Translator', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Sign Language Translator Toolkit')
    sub = parser.add_subparsers(dest='cmd')

    p_collect = sub.add_parser('collect')
    p_collect.add_argument('--output', default='collected_data.csv')
    p_collect.add_argument('--label', required=True)
    p_collect.add_argument('--samples', type=int, default=300)
    p_collect.add_argument('--camera', type=int, default=0)

    p_train = sub.add_parser('train')
    p_train.add_argument('--data', default='collected_data.csv')
    p_train.add_argument('--model', default='model.joblib')
    p_train.add_argument('--test-size', type=float, default=0.2)

    p_run = sub.add_parser('translate')
    p_run.add_argument('--model', default='model.joblib')
    p_run.add_argument('--camera', type=int, default=0)
    p_run.add_argument('--smoothing', type=int, default=5)

    args = parser.parse_args()

    if args.cmd == 'collect':
        collect_data(args.output, args.label, samples=args.samples, camera_id=args.camera)
    elif args.cmd == 'train':
        train_model(args.data, args.model, test_size=args.test_size)
    elif args.cmd == 'translate':
        real_time_translate(args.model, camera_id=args.camera, smoothing=args.smoothing)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
