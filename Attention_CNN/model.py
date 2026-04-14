import tensorflow as tf
from keras import layers, models

def attention_block(x):
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(x.shape[-1] // 8, activation="relu")(attention)
    attention = layers.Dense(x.shape[-1], activation="sigmoid")(attention)
    attention = layers.Reshape((1, 1, x.shape[-1]))(attention)
    return layers.Multiply()([x, attention])

def build_model(num_classes):

    inputs = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = attention_block(x)  # attention added here
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model