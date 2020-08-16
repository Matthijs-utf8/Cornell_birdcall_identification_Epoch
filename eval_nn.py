import argparse

from tensorflow import keras
import tensorflow as tf

import dataloader
from baseline_nn import recall_m, precision_m, f1_m

# if __name__ == '__main__':
#     try:
#         gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#         for device in gpu_devices:
#             tf.config.experimental.set_memory_growth(device, True)
#     except IndexError:
#         pass


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

    test_generator = dataloader.DataGeneratorTestset()
    results = model.evaluate(test_generator)
    results = {out: results[i] for i, out in enumerate(model.metrics_names)}
    print("EVALUATION:")
    padding = max(map(len, results))
    for k,v in results.items():
        print(f"{k:{padding}s}: {v}")
