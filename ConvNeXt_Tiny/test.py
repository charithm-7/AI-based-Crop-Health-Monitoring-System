import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

TEST_DIR = "../dataset_split/test"
MODEL_PATH = "best_model.keras"

IMG_SIZE = (224,224)
BATCH_SIZE = 16

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_data = test_data.map(lambda x,y:(x/255.0,y))

# evaluate
loss, acc = model.evaluate(test_data)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# labels
y_true = np.concatenate([y.numpy() for _,y in test_data])
y_true = np.argmax(y_true, axis=1)

y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

# classification report
report = classification_report(y_true, y_pred)
with open("results/classification_report.txt","w") as f:
    f.write(report)

print(report)

# ROC curve
num_classes = len(np.unique(y_true))
y_true_bin = label_binarize(y_true, classes=range(num_classes))

plt.figure()
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    plt.plot(fpr, tpr)

plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("results/roc_curve.png")
plt.close()

print("Testing completed")