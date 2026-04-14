import tensorflow as tf
from keras import layers
from model import build_model
import os
import matplotlib.pyplot as plt

TRAIN_DIR = "../dataset_split/train"
VAL_DIR   = "../dataset_split/val"

IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 25
SAVE_PATH = "best_model.keras"

# dataset
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

# augmentation
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# normalize
rescale = layers.Rescaling(1./255)

train_data = train_data.map(lambda x,y:(rescale(data_aug(x)),y))
val_data   = val_data.map(lambda x,y:(rescale(x),y))

# model
num_classes = train_data.element_spec[1].shape[-1]
model = build_model(num_classes)

# fine tuning
for layer in model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
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

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# accuracy graph
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.savefig("results/accuracy.png")
plt.close()

# loss graph
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.savefig("results/loss.png")
plt.close()

print("ConvNeXt training done")