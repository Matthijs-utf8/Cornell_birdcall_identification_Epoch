import copy
import glob
import threading

import h5py
import numpy as np
from tensorflow import keras
import pandas as pd
from tensorflow.python.keras.applications.resnet import ResNet50
from tqdm import tqdm

import data_reading
from baseline_preprocess import preprocess, spectrogram_shape, tf_fourier
from birdcodes import bird_code


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_root, batch_size=32, dim=(16, 7, 2048), shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle

        self.data_root = data_root

        # Note: if split into training and test sets, these may not  be the same shape
        self.files = glob.glob(f"{data_root}/*")
        self.indexes = np.arange(len(self.files))

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(files_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_size = min(self.batch_size, len(files_temp))

        X = np.empty((batch_size, *self.dim))
        y = np.zeros((batch_size, len(bird_code)), dtype=int)

        # Generate data
        for i, file in enumerate(files_temp):
            # Store sample

            try:
                data = np.load(file)
                # compressed files are stored as dict
                if type(data) == np.lib.npyio.NpzFile:
                    data = data["arr_0"]

                    # fix shape issue (250, 257) -> (250, 257, 1)
                    data = data[:, :, np.newaxis]

            except ValueError as e:
                raise ValueError("Malformed numpy file: " + file) from e

            try:
                X[i,] = np.reshape(data, self.dim)
            except ValueError as e:
                raise ValueError("Cannot reshape data from file: " + file + ", with shape: " + str(data.shape)) from e

            # Store class
            bird_name = file.split("/")[-1].split("_")[0]
            y[i, bird_code[bird_name]] = 1
        return X, y

    def split(self, factor=0.1):
        """ Split into training and validation sets, probably very not thread safe """
        split = int(len(self.indexes) * (1 - factor))
        train_indices, test_indices = self.indexes[:split], self.indexes[split:]
        test = copy.deepcopy(self)

        self.indexes = train_indices
        test.indexes = test_indices
        return self, test



class DataGeneratorHDF5(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, file, batch_size=32, dim=(16, 7, 2048), shuffle=True, verbose=True):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle

        self.file = file

        self.f = h5py.File(file, "r")
        self.length = len(self.f["spectrograms"])
        if verbose:
            print("Using dataset", file)
            print("  Size", self.length)
            print("  Metadata:")
            for k, v in self.f["images"].attrs.items():
                print("   ", k, v)

        # Note: if split into training and test sets, these may not  be the same shape
        self.indexes = np.arange(len(self.files))

        self.X = self.f["spectrograms"]
        self.y = self.f["labels"]

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.X[indexes]
        y = self.y[indexes]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def split(self, factor=0.1):
        """ Split into training and validation sets, probably very not thread safe """
        split = int(len(self.indexes) * (1 - factor))
        train_indices, test_indices = self.indexes[:split], self.indexes[split:]
        test = copy.deepcopy(self)

        self.indexes = train_indices
        test.indexes = test_indices
        return self, test



class DataGeneratorTestset(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, batch_size=32, use_resnet=False, channel=True):
        """

        Args:
            batch_size:
            use_resnet:
            channel: spectrogram shape if true: (?, 250, 257, 1)  if false: (?, 250, 257)
        """
        self.batch_size = batch_size
        self.channel = channel

        data_root = data_reading.test_data_base_dir + "example_test_audio/"
        label_root = data_reading.test_data_base_dir + "example_test_audio_summary.csv"
        self.labels = pd.read_csv(label_root)
        self.files = glob.glob(f"{data_root}/*")

        self.use_resnet = use_resnet
        if use_resnet:
            self.resnet = ResNet50(input_shape=(spectrogram_shape + (3,)), include_top=False)

        self.__data_generation()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        return self.X[index:index + self.batch_size], self.y[index:index + self.batch_size]

    def __data_generation(self):

        self.X = []
        self.y = []

        for file in self.files:
            file_id = file.split("/")[-1].split("_")[0]
            if self.use_resnet:
                fragments = preprocess(file, self.resnet)
            else:
                fragments = tf_fourier(file, display=True)
                if self.channel:
                    # shape (?, 250, 257) -> (?, 250, 257, 1) aka add channel
                    fragments = fragments[:, :, :, np.newaxis]

            for i, fragment in enumerate(fragments):
                t_start, t_end = i * 5, i * 5 + 5
                self.X.append(fragment)

                rows_file = self.labels["filename"] == file_id
                rows_time = self.labels["seconds"] == t_end
                detected_birds_ecodes = self.labels.loc[rows_file & rows_time]['birds']

                assert len(
                    detected_birds_ecodes) <= 1, "Multiple entries for time segment in test audio summary csv file"
                try:
                    detected_birds_ecodes = detected_birds_ecodes.iloc[0].split(" ")
                    detected_birds = {bird_code.get(x, None) for x in detected_birds_ecodes}
                except:
                    detected_birds = {}

                y = np.array([1 if i in detected_birds else 0 for i in bird_code.values()])
                self.y.append(y)

        if self.use_resnet:
            self.X = np.concatenate(self.X)  # concat from (32, 1, 16, 7, 2048) to (32, 16, 7, 2048)
        self.X = np.array(self.X)
        self.y = np.array(self.y)


if __name__ == '__main__':
    # print("\nTRAIN\n")
    # d = DataGenerator("preprocessed")
    # X, y = d[0]
    # print(X.shape, y.shape)

    print("\nTEST\n")
    d = DataGeneratorTestset()
    X, y = d[0]
    print(X.shape, y.shape)
