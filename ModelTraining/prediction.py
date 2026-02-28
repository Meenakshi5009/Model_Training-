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

IMG_SIZE = 150
model = load_model("gif_video_bullying_model.h5")

# -------------------------
# COMMON PREPROCESS
# -------------------------
def preprocess(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)
# -------------------------
# PREDICTION FUNCTIONS
# -------------------------

def predict_image(img_path):
    img = cv2.imread(img_path)
    return model.predict(preprocess(img), verbose=0)[0][0]
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    preds = []
    frame_count = 0
    max_frames = 50   # Limit total frames (very important)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame only
        if frame_count % 10 == 0:
            preds.append(model.predict(preprocess(frame), verbose=0)[0][0])

        frame_count += 1

        # Stop after max_frames
        if frame_count > max_frames:
            break

    cap.release()
    return np.mean(preds) if preds else 0.0

def predict_gif(gif_path):
    frames = imageio.mimread(gif_path)
    preds = []

    for frame in frames[:10]:  # Only first 10 frames
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        preds.append(model.predict(preprocess(frame), verbose=0)[0][0])

    return np.mean(preds) if preds else 0.0

# -------------------------
# PART 1: EVALUATE ON LABELED TEST DATA
# -------------------------
TEST_DIR = "test_inputs"
y_true = []
y_pred = []

# Folder structure:
# test_inputs/bullying/
# test_inputs/non_bullying/
for label in ["bullying", "non_bullying"]:
    true_label = 0 if label == "bullying" else 1
    folder = os.path.join(TEST_DIR, label)

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
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

        pred_label = 0 if score < 0.5 else 1
        y_true.append(true_label)
        y_pred.append(pred_label)

# -------------------------
# FINAL METRICS + CONFUSION MATRIX
# -------------------------
if y_true and y_pred:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n📊 FINAL EVALUATION METRICS")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")

    print("\n📄 Classification Report:\n")
    print(classification_report(y_true, y_pred,
                                target_names=["Bullying", "Non-Bullying"]))

    # -------------------------
    # CONFUSION MATRIX GRAPH (SECOND CASE)
    # -------------------------
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Bullying", "Non-Bullying"]
    )

    disp.plot()
    plt.title("Confusion Matrix for Cyberbullying Detection")
    plt.show()

else:
    print("No labeled data found for metrics calculation.")

# -------------------------
# PART 2: PREDICT ON UNLABELED NEW INPUTS
# -------------------------
NEW_INPUT_DIR = "inputs"
if os.path.exists(NEW_INPUT_DIR):
    print("\n🔹 PREDICTIONS ON NEW UNLABELED INPUTS\n")
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
            print(f"Skipped unsupported file: {file}")
            continue

        pred_label = "Bullying" if score < 0.5 else "Non-Bullying"
        print(f"File: {file} --> Predicted: {pred_label} (Score: {score:.4f})")
else:
    print("No new input folder found.")
