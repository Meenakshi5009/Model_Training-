import os
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 150
THRESHOLD = 0.6   # You can adjust after threshold testing

model = load_model("models/gif_video_bullying_model.h5")

# -------------------------
# PREPROCESS FUNCTION
# -------------------------
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

# -------------------------
# PREDICTION FUNCTIONS
# -------------------------
def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0.0
    return model.predict(preprocess(img), verbose=0)[0][0]

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    preds = []
    frame_count = 0
    max_frames = 50

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            preds.append(model.predict(preprocess(frame), verbose=0)[0][0])

        frame_count += 1
        if frame_count > max_frames:
            break

    cap.release()
    return np.mean(preds) if preds else 0.0

def predict_gif(gif_path):
    frames = imageio.mimread(gif_path)
    preds = []

    for frame in frames[:10]:
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        preds.append(model.predict(frame, verbose=0)[0][0])

    return np.mean(preds) if preds else 0.0

# -------------------------
# EVALUATE TEST DATA
# -------------------------
TEST_DIR = "test_inputs"
y_true = []
y_pred = []
scores = []   # 🔥 store raw scores

for label in ["bullying", "non_bullying"]:
    true_label = 0 if label == "bullying" else 1
    folder = os.path.join(TEST_DIR, label)

    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        ext = os.path.splitext(file)[1].lower()

        if ext in [".jpg", ".jpeg", ".png"]:
            score = predict_image(path)
        elif ext == ".gif":
            score = predict_gif(path)
        elif ext == ".mp4":
            score = predict_video(path)
        else:
            continue

        scores.append(score)

        pred_label = 0 if score < THRESHOLD else 1
        y_true.append(true_label)
        y_pred.append(pred_label)

# -------------------------
# METRICS IN %
# -------------------------
if y_true:

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred) * 100
    rec = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100

    print("\n📊 FINAL METRICS (IN %)")
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Precision : {prec:.2f}%")
    print(f"Recall    : {rec:.2f}%")
    print(f"F1-Score  : {f1:.2f}%")

    print("\n📄 Classification Report:\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Bullying", "Non-Bullying"]
    ))

    # -------------------------
    # THRESHOLD TESTING
    # -------------------------
    print("\n🔎 Threshold Testing:\n")

    for t in [0.4, 0.5, 0.6, 0.7]:
        temp_preds = [0 if s < t else 1 for s in scores]
        temp_acc = accuracy_score(y_true, temp_preds) * 100
        print(f"Threshold {t} → Accuracy: {temp_acc:.2f}%")

# -------------------------
# NEW INPUT PREDICTIONS
# -------------------------
NEW_INPUT_DIR = "inputs"

if os.path.exists(NEW_INPUT_DIR):
    print("\n🔹 PREDICTIONS ON NEW INPUTS\n")

    for file in os.listdir(NEW_INPUT_DIR):
        path = os.path.join(NEW_INPUT_DIR, file)
        ext = os.path.splitext(file)[1].lower()

        if ext in [".jpg", ".jpeg", ".png"]:
            score = predict_image(path)
        elif ext == ".gif":
            score = predict_gif(path)
        elif ext == ".mp4":
            score = predict_video(path)
        else:
            continue

        print(f"Score: {score:.4f}")

        if score < THRESHOLD:
            pred_label = "Bullying"
        else:
            pred_label = "Non-Bullying"

        print(f"File: {file} → Predicted: {pred_label}\n")
