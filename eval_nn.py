import argparse

from tensorflow import keras
import tensorflow as tf

import dataloader
from baseline_nn import recall_m, precision_m, f1_m

if __name__ == '__main__':
    try:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except IndexError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model-file", type=str, help="Model hdf5 file")
    args = parser.parse_args()

    model = keras.models.load_model(args.model_file, custom_objects={
        'recall_m': recall_m,
        'precision_m': precision_m,
        'f1_m': f1_m
    })

    test_generator = dataloader.DataGeneratorTestset()
    results = model.evaluate(test_generator, return_dict=True)
    print("EVALUATION:")
    padding = max(map(len, results))
    for k,v in results.items():
        print(f"{k:{padding}s}: {v}")
