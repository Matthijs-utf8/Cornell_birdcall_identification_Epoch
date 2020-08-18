from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.applications.resnet import ResNet50

import birdcodes

spectrogram_dim = (250, 257)


def CNN():
    input_shape = spectrogram_dim + (1,)

    model = keras.models.Sequential([
        layers.Conv2D(16, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPool2D(),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.Flatten(),
        layers.Dense(len(birdcodes.bird_code), activation="sigmoid"),
    ])
    return model

def ResNetHead():
    input_shape = (8, 9, 2048)

    model = keras.models.Sequential([
        layers.GlobalMaxPool2D(input_shape=input_shape),
        layers.Dense(1024),
        layers.Dense(len(birdcodes.bird_code), activation="sigmoid"),
    ])

    return model

def Conv1D():
    model = keras.models.Sequential([
        layers.Conv1D(256, 3, activation="relu"),
        layers.MaxPool1D(2),
        layers.Conv1D(256, 3, activation="relu"),
        layers.MaxPool1D(2),
        layers.Conv1D(256, 3, activation="relu"),
        layers.MaxPool1D(2),
        layers.Conv1D(256, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(len(birdcodes.bird_code), activation="sigmoid")
    ])
    return model

def ResNet():
    input_shape = spectrogram_dim + (3,)
    model = keras.models.Sequential([
        ResNet50(input_shape=input_shape, include_top=False, weights=None),
        layers.GlobalMaxPool2D(input_shape=(8, 9, 2048)),
        layers.Dense(1024),
        layers.Dense(len(birdcodes.bird_code), activation="sigmoid"),
    ])
    return model

