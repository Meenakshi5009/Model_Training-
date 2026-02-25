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
    confusion_matrix
)

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 150
THRESHOLD = 0.35

MODEL_PATH = "gif_video_bullying_model.h5"
TEST_DIR = "test_inputs"        # labeled test folder
NEW_INPUT_DIR = "inputs" # unlabeled data

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model(MODEL_PATH)

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

# -------------------------
# PREDICTION FUNCTIONS
# -------------------------
def predict_image(path):
    img = cv2.imread(path)
    return model.predict(preprocess(img), verbose=0)[0][0]

def predict_video(path):
    cap = cv2.VideoCapture(path)
    preds = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        preds.append(model.predict(preprocess(frame), verbose=0)[0][0])
    cap.release()
    return np.mean(preds) if preds else 0.0

def predict_gif(path):
    frames = imageio.mimread(path)
    preds = []
    for frame in frames[:10]:
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        preds.append(model.predict(preprocess(frame), verbose=0)[0][0])
    return np.mean(preds) if preds else 0.0

# =====================================================
# PART 1: MODEL EVALUATION (LABELED DATA)
# =====================================================
y_true = []
y_pred = []

print("\n📊 Evaluating model on labeled test data...\n")

for label in ["bullying", "non_bullying"]:
    true_label = 0 if label == "bullying" else 1
    folder = os.path.join(TEST_DIR, label)

    if not os.path.exists(folder):
        print(f"⚠ Folder missing: {folder}")
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

        pred_label = 0 if score < THRESHOLD else 1
        y_true.append(true_label)
        y_pred.append(pred_label)

# -------------------------
# METRICS
# -------------------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("✅ FINAL METRICS")
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {prec*100:.2f}%")
print(f"Recall    : {rec*100:.2f}%")
print(f"F1-Score  : {f1*100:.2f}%\n")

print("📄 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Bullying", "Non-Bullying"]))

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Bullying", "Non-Bullying"])
plt.yticks([0,1], ["Bullying", "Non-Bullying"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------
# METRICS BAR GRAPH
# -------------------------
plt.figure()
metrics = [acc, prec, rec, f1]
labels = ["Accuracy", "Precision", "Recall", "F1-score"]
plt.bar(labels, metrics)
plt.ylim(0, 1)
plt.title("Model Performance Metrics")
plt.show()

# =====================================================
# PART 2: PREDICTION ON UNLABELED DATA
# =====================================================
print("\n🔹 Predictions on NEW unlabeled data\n")

if os.path.exists(NEW_INPUT_DIR):
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

        label = "Bullying" if score < THRESHOLD else "Non-Bullying"
        print(f"{file} → {label} (score = {score:.4f})")
else:
    print("❌ No unlabeled input folder found.")
