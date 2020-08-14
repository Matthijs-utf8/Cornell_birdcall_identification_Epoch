import datetime

import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from multiprocessing import Process

from tqdm import tqdm
import numpy as np
from scipy.signal import resample
import librosa
from matplotlib import pyplot as plt

from Noise_Extractor import filter_sound, get_frames
import data_reading
import argparse

from birdcodes import bird_code

window_size = 440
universal_sample_rate = 22000
spectrogram_slices_per_input = universal_sample_rate * 5 // window_size # = 5 seconds

if __name__ == '__main__':

    try:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except IndexError:
        pass

def preprocess(file_path, feature_extractor: keras.models.Model):
    """
    Loads the audio file, generates a spectrogram, and applies feature_extractor on it
    """
    spectrograms = tf_fourier(file_path)

    if spectrograms != []:
        # Duplicate the single amplitude channel to 3 channels, because ResNet50 expects 3 channels
        spectrograms = np.array(spectrograms)
        spectrograms = np.reshape(spectrograms, spectrograms.shape + (1,))
        spectrograms = np.repeat(spectrograms, 3, axis=3)
        
        # Apply the feature extractor
        return feature_extractor.predict(spectrograms)
    else:
        return []

def tf_fourier(file_path):
    """
    Loads the audio file, and applies the short-time fourier transform implemented on the GPU by TensorFlow
    """
    try:
        sound, sample_rate = librosa.load(file_path)
    except ZeroDivisionError as e:
        raise ZeroDivisionError("File for error above:", file_path) from e

    # Make sure all files have the same sample rate
    if sample_rate != universal_sample_rate:
        sound = resample(sound, int(universal_sample_rate * (len(sound) / sample_rate)))
        pass
    
    # Generate the spectrogram
    spectrogram = tf.abs(
        tf.signal.stft(tf.reshape(sound, [1, -1]), window_size, window_size)
    )[0]

    # Split up into slices of (by default) 5 seconds
    slices = [
        spectrogram[
             i      * spectrogram_slices_per_input :
            (i + 1) * spectrogram_slices_per_input
        ]
        for i in range(spectrogram.shape[0] // spectrogram_slices_per_input)
    ]

    return np.array(slices)

spectrogram_shape = (250, 257, 1)
DATASET_VERSION = "1.0.0"


if __name__ == "__main__":
    import sys
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_mode", default="spectrogram", type=str, help="Possible values: 'spectrogram' or 'resnet'")
    parser.add_argument("--dir", default="preprocessed.h5", type=str, help="Where to place the hdf5 dataset file")
    parser.add_argument("--info", type=str, help="Description to add to hdf5 file metadata")
    parser.add_argument("-b", "--bird_codes", nargs="*", default=[], type=str, help="List of birdcodes indicating which files need to be processed")
    args = parser.parse_args()


    
    output_dir = args.dir
    use_resnet = args.feature_mode == "resnet"
    if use_resnet:
        raise NotImplementedError("HDF5 not set up for this, and naming scheme is incorrect (And breaking changes to shape and such)")

    print("All birdcodes to process:", " ".join(args.bird_codes))

    # if use_resnet:
    #     resnet: keras.models.Model = ResNet50(input_shape=(spectrogram_shape + (3,)), include_top=False)
    


    # Process all files based on the birdcodes in the arguments
    if args.bird_codes == []:
        args.bird_codes = bird_code.keys()

    i = 0
    with h5py.File(args.dir, "w") as f:

        for birdcode in tqdm(args.bird_codes):
            print(birdcode)
            bird_id = bird_code[birdcode]

            path_to_birdsound_dir = data_reading.test_data_base_dir + "train_audio/" + birdcode + "/"

            for file_name in os.listdir(path_to_birdsound_dir):
                # if use_resnet:
                #     fragments = preprocess(path_to_birdsound_dir + file_name, resnet)
                # else:
                fragments = tf_fourier(path_to_birdsound_dir + file_name)

                # shape (?, 250, 257) -> (?, 250, 257, 1) aka add channel
                fragments = fragments[:, :, :, np.newaxis]

                # match number of labels to fragments
                labels = np.array([bird_id] * len(fragments))
                print("Shape", fragments.shape)
                print("Shape label", labels.shape)

                if "spectrograms" not in f:
                    dataset = f.create_dataset(
                        "spectrograms", np.shape(fragments), np.float32, maxshape=(None,) + spectrogram_shape,
                        data=fragments, chunks=True,
                        # compression="gzip"
                    )
                    label_set = f.create_dataset(
                        "labels", np.shape(labels), np.int, maxshape=(None,), data=labels, chunks=True
                    )
                else:
                    shape = np.array(dataset.shape)
                    shape[0] += fragments.shape[0]
                    dataset.resize(shape)
                    dataset[-fragments.shape[0]:, ...] = fragments

                    shape = np.array(label_set.shape)
                    shape[0] += labels.shape[0]
                    label_set.resize(shape)
                    label_set[-labels.shape[0]:, ...] = labels

            i += 1
            if i == 3:
                break

        dataset.attrs["version"] = DATASET_VERSION
        dataset.attrs["feature_mode"] = args.feature_mode
        dataset.attrs["info"] = args.info
        dataset.attrs["creation"] = datetime.datetime.now()
        dataset.attrs["bird_code"] = bird_code


