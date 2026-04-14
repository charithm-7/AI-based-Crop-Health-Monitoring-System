import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

TEST_DIR = "../dataset_split/test"
MODEL_PATH = "best_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# normalize
test_data = test_data.map(lambda x, y: (x / 255.0, y))

# true labels
y_true = np.concatenate([y.numpy() for _, y in test_data])
y_true = np.argmax(y_true, axis=1)

# predictions
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.close()

print("Attention CNN testing completed")