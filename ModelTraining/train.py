import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 150
BATCH_SIZE = 16
EPOCHS = 35

TRAIN_DIR = r"C:\Users\meena\OneDrive\Desktop\ModelTrain\train"
VAL_DIR   = r"C:\Users\meena\OneDrive\Desktop\ModelTrain\val"

# -------------------------
# DATA GENERATORS
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    shear_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# -------------------------
# SAFETY CHECK
# -------------------------
print("Train samples:", train_gen.samples)
print("Validation samples:", val_gen.samples)
print("Class mapping:", train_gen.class_indices)

if train_gen.samples == 0 or val_gen.samples == 0:
    raise ValueError("❌ No images found. Check folder structure.")

# -------------------------
# BASE MODEL
# -------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze all layers initially
base_model.trainable = False

# -------------------------
# CUSTOM HEAD
# -------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------
# COMPILE
# -------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# CLASS WEIGHTS
# -------------------------
class_weight = {
    0: 2.0,  # bullying
    1: 1.0   # non-bullying
}

# -------------------------
# TRAIN INITIAL MODEL
# -------------------------
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop]
)

# -------------------------
# OPTIONAL: FINE-TUNE LAST LAYERS
# -------------------------
# Unfreeze last 20 layers of base_model
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Compile again with smaller learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Continue training for a few more epochs
history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,  # fine-tune 10 epochs
    class_weight=class_weight,
    callbacks=[early_stop]
)

# -------------------------
# SAVE MODEL
# -------------------------
os.makedirs("models", exist_ok=True)
model.save("models/gif_video_bullying_model.h5")
print("✅ Model trained and saved successfully")
