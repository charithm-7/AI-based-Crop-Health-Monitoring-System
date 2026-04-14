import tensorflow as tf
from keras import layers
from model import build_model
import matplotlib.pyplot as plt
import os

TRAIN_DIR = "../dataset_split/train"
VAL_DIR   = "../dataset_split/val"

IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 20
SAVE_PATH = "best_model.keras"

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

# normalization
rescale = layers.Rescaling(1./255)
train_data = train_data.map(lambda x,y:(rescale(x),y))
val_data   = val_data.map(lambda x,y:(rescale(x),y))

num_classes = train_data.element_spec[1].shape[-1]
model = build_model(num_classes)

# fine tuning (important for higher accuracy)
for layer in model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

os.makedirs("results",exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint,earlystop]
)

# Accuracy graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train","val"])
plt.savefig("results/accuracy.png")
plt.close()

# Loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train","val"])
plt.savefig("results/loss.png")
plt.close()

print("Training finished")