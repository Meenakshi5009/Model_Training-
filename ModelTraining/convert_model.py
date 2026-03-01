import tensorflow as tf

model = tf.keras.models.load_model("gif_video_bullying_model.h5", compile=False)
model.save("gif_video_bullying_model.keras")

print("✅ Model converted successfully!")