import copy
import glob
import threading

import h5py
import numpy as np
from tensorflow import keras
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tqdm import tqdm
import torch

import data_reading
# from baseline_preprocess import preprocess, spectrogram_shape
from birdcodes import bird_code
import preprocessing


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_root, batch_size=32, dim=(16, 7, 2048), shuffle=True, channels=1):
        """
        Datagenerator for the training set from a folder of preprocessed numpy files containing spectrograms
        :param data_root: The data folder
        :param batch_size: Training batch size
        :param dim: The input dimensions
        :param shuffle: Whether to shuffle the data each epoch
        :param channels: The number of channels to add to the data eiter 0 (no channels), 1 or 3
        """
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.channels = channels

        self.data_root = data_root

        # Note: if split into training and test sets, these may not  be the same shape
        self.files = glob.glob(f"{data_root}/*")
        assert len(self.files), f"{data_root} not found"
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

        assert len(X), f"Empty batch {len(self)} {index} {index * self.batch_size}"

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

                    if self.channels == 3:
                        # shape (250, 257) -> (250, 257, 3) aka add channels
                        data = np.repeat(data[:, :, np.newaxis], 3, -1)
                    elif self.channels == 1:
                        # shape (250, 257) -> (250, 257, 1) aka add channel
                        data = data[:, :, np.newaxis]


            except ValueError as e:
                raise ValueError("Malformed numpy file: " + file) from e

            try:
                # X[i,] = np.reshape(data, self.dim)
                X[i,] = data
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

    def __init__(self, filename, batch_size=32, dim=(16, 7, 2048), shuffle=True, verbose=True):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.verbose = verbose

        self.filename = filename
        assert ".hdf5" in filename, "Not hdf5 file"

    def __enter__(self):
        self.file = h5py.File(self.filename, "r")

        self.length = len(self.file["data"])
        if self.verbose:
            print("Using dataset", self.filename)
            print("  Number of samples", self.length)
            print("  Metadata:")
            for k, v in self.file["data"].attrs.items():
                print("   ", k, v)

        self.X = self.file["data"]
        self.y = self.file["labels"]

        # Note: if split into training and test sets, these may not  be the same shape
        self.indexes = np.arange(len(self.X))

        self.on_epoch_end()

        return self

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        indexes = sorted(indexes) # required for hdf5 indexing

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    @staticmethod
    def from_multiple():
        raise NotImplementedError()
        file_names_to_concatenate = ['1.h5', '2.h5', '3.h5', '4.h5']
        entry_key = 'data'  # where the data is inside of the source files.

        sources = []
        total_length = 0
        for i, filename in enumerate(file_names_to_concatenate):
            with h5py.File(file_names_to_concatenate[i], 'r') as activeData:
                vsource = h5py.VirtualSource(activeData[entry_key])
                total_length += vsource.shape[0]
                sources.append(vsource)

        layout = h5py.VirtualLayout(shape=(total_length,),
                                    dtype=np.float)

        offset = 0
        for vsource in sources:
            length = vsource.shape[0]
            layout[offset: offset + length] = vsource
            offset += length

        with h5py.File("VDS_con.h5", 'w', libver='latest') as f:
            f.create_virtual_dataset(entry_key, layout, fillvalue=0)

from torch.utils.data import Dataset, DataLoader

class DataGeneratorHDF5Pytorch(Dataset):
    """Generates data for Keras
    Use in combination with something like torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    """

    def __init__(self, filename, batch_size=32, dim=(16, 7, 2048), shuffle=True, verbose=True):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.verbose = verbose

        self.filename = filename
        assert ".hdf5" in filename, "Not hdf5 file"

    def __enter__(self):
        self.file = h5py.File(self.filename, "r")

        self.length = len(self.file["data"])
        if self.verbose:
            print("Using dataset", self.filename)
            print("  Number of samples", self.length)
            print("  Metadata:")
            for k, v in self.file["data"].attrs.items():
                print("   ", k, v)

        self.X = self.file["data"]
        self.y = self.file["labels"]

        # Note: if split into training and test sets, these may not  be the same shape
        self.indexes = np.arange(len(self.X))
        return self

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        indexes = sorted(indexes) # required for hdf5 indexing

        X = self.X[indexes]
        y = self.y[indexes]

        return torch.from_numpy(X), torch.from_numpy(y)

    def split(self, factor=0.1):
        """ Split into training and validation sets, probably very not thread safe """
        split = int(len(self.indexes) * (1 - factor))
        train_indices, test_indices = self.indexes[:split], self.indexes[split:]
        test = copy.deepcopy(self)

        self.indexes = train_indices
        test.indexes = test_indices
        return self, test

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

class DataGeneratorTestset(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, batch_size=32, use_resnet=False, channels=1, normalize_samples=False, filter_noise=False):
        """
        Create dataloader for Cornell test data.
        Args:
            batch_size:
            use_resnet:
            channels: spectrogram shape if true: (?, 257, 463, channels)  if false: (?, 257, 463)
            normalize_samples: bool, indicates whether loaded test audio should be normalized
            filter_noise: bool, indicates whether loaded test audio should be filtered for noise
        """
        self.batch_size = batch_size
        self.channels = channels
        self.normalize_samples = normalize_samples
        self.filter_noise = filter_noise

        data_root = data_reading.test_data_base_dir + "example_test_audio/"
        label_root = data_reading.test_data_base_dir + "example_test_audio_summary.csv"
        self.labels = pd.read_csv(label_root)
        self.files = glob.glob(f"{data_root}/*")

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
            fragments = preprocessing.load_spectrograms(file, self.normalize_samples, self.filter_noise)

            if self.channels == 3:
                fragments = np.repeat(fragments[:, :, :, np.newaxis], 3, -1)
            elif self.channels == 1:
                # shape (?, 250, 257) -> (?, 250, 257, 1) aka add channel
                fragments = fragments[:, :, :, np.newaxis]
            elif self.channels in [0, False, None]:
                pass
            else:
                raise NotImplementedError("Invalid channel")

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

        self.X = np.array(self.X)
        self.y = np.array(self.y)


if __name__ == '__main__':
    # print("\nTRAIN\n")
    # d = DataGenerator("preprocessed")
    # X, y = d[0]
    # print(X.shape, y.shape)

    # print("Test HDF5 data generator")
    # with DataGeneratorHDF5("test.hdf5") as ds:
    #     X, y = ds[0]
    #     print(X.shape, y.shape)

    # print("Test Testdata generator")
    # d = DataGeneratorTestset(filter_noise=False, normalize_samples=True)
    # X, y = d[0]
    # print(X.shape, y.shape)
    pass

""" Librosa (without resampling) vs HDF5 () benchmark"""
if __name__ == '__main__':
    import time
    import os
    import librosa
    print("Speed comparison")
    N = 2000

    print("Test HDF5 data generator")

    # start = time.perf_counter()
    # 
    # with DataGeneratorHDF5("datasets/original_small.hdf5", verbose=True, batch_size=1) as ds:
    #     for i in range(N):
    #         X, y = ds[i]
    #
    # duration = time.perf_counter() - start
    # print("Duration", duration)
    # print(duration/N, "seconds per sample\n")

    print("Test librosa load")
    start = time.perf_counter()

    i = 0
    for birdcode in bird_code.keys():
        path_to_birdsound_dir = data_reading.test_data_base_dir + "train_audio/" + birdcode + "/"
        for file_name in os.listdir(path_to_birdsound_dir):
            if i > N:
                break
            sound, sample_rate = librosa.load(path_to_birdsound_dir + file_name, sr=22500)
            nr_of_seconds = len(sound) / sample_rate
            nr_of_frames = nr_of_seconds//5
            i += nr_of_frames

    duration = time.perf_counter() - start
    print("Duration", duration)
    print(duration/N, "seconds per sample\n")