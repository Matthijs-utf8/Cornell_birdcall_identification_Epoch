# imports
import datetime
import time

import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from multiprocessing import Process
import pandas as pd

from tqdm import tqdm
import numpy as np
from scipy.signal import resample
import librosa
from matplotlib import pyplot as plt
import random

from Noise_Extractor import filter_sound, get_frames
import data_reading
import argparse
import sound_shuffling
import preprocessing

from birdcodes import bird_code

# variables
universal_sample_rate = 22000
window_size = 440
base_dir = data_reading.read_config()
df_train = pd.read_csv(base_dir + "train.csv")


class HDF5DatasetExtendable:
    VERSION = "1.0.0"

    def __init__(self, filename, data_type=np.float32, label_type=np.int, compression=None):
        """
        Set initial parameters for auto-resizable hdf5 dataset
        :param filename: The filenamae
        :param data_type: Datatype to use for storage, such as np.float32
        :param label_type: Datatype to use for storage, such as np.float32
        :param compression: None or "gzip"
        """
        assert ".hdf5" in filename, "Filename ust be .hdf5 file"
        self.filename = filename
        self.data_type = data_type
        self.label_type = label_type
        self.compression = compression
        self.initialized = False

    def __enter__(self):
        self.file = h5py.File(self.filename, "w")
        return self

    def add_metadata(self, info):
        """
        Add metedata to the dataset attributes. This data can be displayed when starting training, and
        informs the user of what this dataset contains. Be descriptive!
        Tip: a good starting point is just pass vars(args), such that all commandline options are logged.
        :param info: A dictionary with user information such as {"augmentation":"shifted"}
        """
        self.dataset.attrs["version"] = self.VERSION
        for k, v in info.items():
            self.dataset.attrs[k] = str(v)

    def _init(self, data, labels):
        """
        Initialize the dataset objects with the first batch of data.
        Do not call this function, always call append.
        :param data: numpy array containing the data, where the first axis is the sample index
        :param labels: numpy array containing the labels, where the first axis is the sample index
        """
        self.dataset = self.file.create_dataset(
            "data", np.shape(data), self.data_type, maxshape=(None,) + np.shape(data)[1:],
            data=data, chunks=True,
            compression=self.compression
        )
        self.labelset = self.file.create_dataset(
            "labels", np.shape(labels), self.label_type, maxshape=(None,) + np.shape(labels)[1:],
            data=labels, chunks=True,
            compression=self.compression
        )
        self.initialized = True

    def append(self, data, labels):
        """
        Add data to the dataset. If not initialized, this will copy the shapes from the first call and initialze.
        :param data: numpy array containing the data, where the first axis is the sample index
        :param labels: numpy array containing the labels, where the first axis is the sample index
        """
        if not self.initialized:
            self._init(data, labels)
            return

        shape = np.array(self.dataset.shape)
        shape[0] += data.shape[0]
        self.dataset.resize(shape)
        self.dataset[-data.shape[0]:, ...] = data

        shape = np.array(self.labelset.shape)
        shape[0] += labels.shape[0]
        self.labelset.resize(shape)
        self.labelset[-labels.shape[0]:, ...] = labels

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


def preprocess(file_path, args):
    # Get sound and sample rate from file using librosa
    try:
        sound, sample_rate = librosa.load(file_path)
    except ZeroDivisionError as e:
        raise ZeroDivisionError("File for error above:", file_path) from e

    # Resampling
    if sample_rate != universal_sample_rate:
        sound = resample(sound, int(universal_sample_rate * (len(sound) / sample_rate)))
        pass

    # If argument for noise addition is set, adds random white- or background noise or removes noise
    if args.noise_aug:
        if args.noise_aug == "white_noise":
            if args.n_steps:
                sound = sound_shuffling.add_white_noise(sound, target_snr=np.random.normal(args.n_steps[0], args.n_steps[1]))
            else:
                sound = sound_shuffling.add_white_noise(sound, target_snr=np.random.normal(4.5, 2.0))
        if args.noise_aug == "background_noise":
            sound = sound_shuffling.add_random_background_noise(sound, sample_rate)
        if args.noise_aug == "no_noise":
            sound = preprocessing.extract_noise(sound, sample_rate, window_width=2048, step_size=512, verbose=False)

    # If argument for shifting is set, shifts amplitude, frequency or time randomly
    if args.shift_aug:
        if args.shift_aug == "amplitude_shift":
            n_steps = random.randint(0, 5)
            sound = sound_shuffling.amplitude_shift(sound, n_steps)
        if args.shift_aug == "frequency_shift":
            n_steps = random.randint(-5, 5)
            sound = sound_shuffling.frequency_shift(sound, sample_rate, n_steps)
        if args.shift_aug == "time_stretch":
            n_steps = random.randint(-5, 5)
            sound = sound_shuffling.time_stretch(sound, n_steps)

    # Normalize
    sound = preprocessing.normalize(sound)

    # Cut sound up in frames of 5 seconds
    window_width = universal_sample_rate * 5
    step_size = window_width  # TODO: paramererize stepsize
    nr_of_frames, frames = get_frames(sound, window_width, step_size)

    return np.array(frames)


def create_data(path_to_birdsound_dir, file_name, args):
    # Get fragments (raw sound files)
    fragments = preprocess(path_to_birdsound_dir + file_name, args)
    # print('fragments shape', fragments.shape)

    # Match number of labels to fragments
    labels = np.array([[1 if i in [bird_id] else 0 for i in bird_code.values()]] * len(fragments))  # one hot encoding
    # print('labels shape', labels.shape)

    return fragments, labels


""" A function that uses a few methods from sound_shuffling.py to be able to easily create a new shuffled dataset. """
def create_shuffled_dataset(nr_of_files, files_to_combine, metrics=[], clip_seconds=5):
    """
    :param nr_of_files: int, number of 5 second files that should be created
    :param metrics: list (or empty list), common metrics that the birdsounds should have
    :param files_to_combine: int, number of birds to overlap eachother
    :param clip_seconds: int, number of seconds that the new clip should be
    """

    for _ in range(nr_of_files):

        # Get the sorted dataframe and pick random files from it (if metrics == None, the files will be randomly
        # picked from the whole dataset)
        new_dataframe = sound_shuffling.filter_metadata_by_metrics(df_train, metrics=metrics,
                                                                   nr_of_files=files_to_combine)
        random_files = sound_shuffling.pick_files_at_random(new_dataframe, nr_of_files=files_to_combine)

        # Combine the files. Normalization happens in this step as well.
        combined_file, labels = sound_shuffling.combine_files(files=random_files,
                                                              universal_sr=22050,
                                                              seconds=clip_seconds)

        return combined_file, labels


def create_shuffled_data(args):
    print(int(args.shuffle_aug[0]), int(args.shuffle_aug[1]))
    print(args.metric)
    create_shuffled_dataset(int(args.shuffle_aug[0]), int(args.shuffle_aug[1]), list(args.metric))

if __name__ == "__main__":
    import sys
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="preprocessed.h5", type=str, help="Where to place the hdf5 dataset file")
    parser.add_argument("--info", type=str, help="Description to add to hdf5 file metadata")
    parser.add_argument("--compression", type=str, help="HDF5 compression algorithm, [None, 'gzip', 'lzf'] ")
    parser.add_argument("-b", "--bird_codes", nargs="*", default=list(bird_code.keys()), type=str,
                        help="List of bird codes indicating which files need to be processed")
    parser.add_argument("--shift_aug", type=str, default=None,
                        help="Possible values: 'amplitude_shift, 'frequency_shift' or 'time_stretch'")
    parser.add_argument("--noise_aug", type=str, default=None,
                        help="Possible values: 'white_noise', 'background_noise', 'no_noise' ")
    parser.add_argument("--shuffle_aug", type=str, default=None, nargs='*', help="Number of files <space> files to "
                                                                                 "combine")
    parser.add_argument("-m", "--metric", type=str, default=None, nargs="*", help="Metrics to combine, separated with "
                                                                                  "spaces, use with --shuffle_aug")
    parser.add_argument("-n", "--n_steps", type=float, nargs="*", help="Randomness noise, use with --noise_aug "
                                                                       "'white_noise', 2 floats")
    parser.add_argument("--max_size", type=int, default=None, help="Maximum number of soundfiles per bird, usefull to limit dataset size.")
    args = parser.parse_args()

    # Print bird code-processing information for user
    if args.bird_codes == list(bird_code.keys()):
        print("All bird codes are processed")
    else:
        print("All bird codes to process:", " ".join(args.bird_codes))

    # Create the dataset
    with HDF5DatasetExtendable(args.file, compression=args.compression) as dataset:

        # Special case: shuffled data
        if args.shuffle_aug:
            data, labels = create_shuffled_dataset(int(args.shuffle_aug[0]), int(args.shuffle_aug[1]), list(args.metric))
            dataset.append(data, labels)

        # Normal cases
        else:
            # Loop over all birds and all files
            for birdcode in tqdm(args.bird_codes):

                bird_id = bird_code[birdcode]
                path_to_birdsound_dir = data_reading.test_data_base_dir + "train_audio/" + birdcode + "/"

                for i, file_name in enumerate(os.listdir(path_to_birdsound_dir)):
                    # limit dataset file by including limited number of sound files
                    if i == args.max_size:
                        break

                    # Create data per file
                    data, labels = create_data(path_to_birdsound_dir, file_name, args)
                    if len(data) == 0:
                        print("Skipping short sound file: ", file_name)
                        continue
                    dataset.append(data, labels)

        # store every argument of the script with the dataset for reproducability
        dataset.add_metadata(vars(args))
