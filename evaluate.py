import argparse
import time

from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

import dataloader
# from utils import recall_m, precision_m, f1_m
import birdcodes

# if __name__ == '__main__':
#     try:
#         gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#         for device in gpu_devices:
#             tf.config.experimental.set_memory_growth(device, True)
#     except IndexError:
#         pass

# confidence_boost = 1.5
confidence_boost = 1


def recall_m(y_true, y_pred):
    y_pred *= confidence_boost
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_pred *= confidence_boost
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="hdf5 model file")
    args = parser.parse_args()

    print("Loading model", args.model)
    model = keras.models.load_model(args.model, custom_objects={
        'recall_m': recall_m,
        'precision_m': precision_m,
        'f1_m': f1_m
    })

    input_shape = (250, 257, 3)

    start = time.time()
    bsize = 32
    channels = 3 #1 # or 3
    test_generator = dataloader.DataGeneratorTestset(channels=channels, batch_size=bsize)
    _, train_generator = dataloader.DataGenerator("spectrograms", batch_size=bsize, dim=input_shape, channels=channels, shuffle=False).split(0.003)

    # switch here
    generator = test_generator


    print("Dataloader done, time (s):", time.time() - start)
    print("Evaluating:")

    prediction = model.predict(generator)

    count = 0
    for i, p in enumerate(prediction):
        labels = generator[i//bsize][1][i%bsize] # select batch, select labels, select sample
        label_idx = np.where(labels)[0]

        p *= confidence_boost
        count += K.sum(K.round(K.clip(p, 0, 1)))
        predicted_positives = K.round(K.clip(p, 0, 1)).numpy().astype(np.int)
        idx, = np.where(predicted_positives)
        predicted_birds = [birdcodes.inverted_bird_code[m] for m in idx]

        print("prediciton", i, predicted_birds, "prob", p[idx], "label", [birdcodes.inverted_bird_code[x] for x in label_idx])


    print("Number of predicted positives", count.numpy())



    results = model.evaluate(generator)
    results = {out: results[i] for i, out in enumerate(model.metrics_names)}
    print("EVALUATION:")
    padding = max(map(len, results))
    for k,v in results.items():
        print(f"{k:{padding}s}: {v}")
