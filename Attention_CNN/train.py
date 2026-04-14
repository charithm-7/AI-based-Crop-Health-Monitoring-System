import tensorflow as tf
from model import build_model
import os
import matplotlib.pyplot as plt

TRAIN_DIR = "../dataset_split/train"
VAL_DIR   = "../dataset_split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SAVE_PATH = "best_model.keras"

# load dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# normalize
rescale = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (rescale(x), y))
val_data   = val_data.map(lambda x, y: (rescale(x), y))

# build model
num_classes = train_data.element_spec[1].shape[-1]
model = build_model(num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("results", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# accuracy plot
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.savefig("results/accuracy.png")
plt.close()

# loss plot
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.savefig("results/loss.png")
plt.close()

print("Attention CNN training completed")