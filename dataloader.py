import glob
import numpy as np
from tensorflow import keras
import pandas as pd

import data_reading
from baseline_preprocess import preprocess, feature_extractor
from birdcodes import bird_code


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_root, batch_size=32, dim=(16,7,2048), shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle

        self.data_root = data_root
        self.files = glob.glob(f"{data_root}/*")
        self.indexes = np.arange(len(self.files))

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

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
                X[i,] = np.reshape(np.load(file), self.dim)
            except ValueError as e:
                raise ValueError("Malformed numpy file:" + file) from e

            # Store class
            bird_name = file.split("/")[-1].split("_")[0]
            y[i, bird_code[bird_name]] = 1
        return X, y# keras.utils.to_categorical(y, num_classes=len(bird_code))

def compute_overlap(x1, y1, x2, y2):
    return max(x1,y1) - min(x2,y2)

# minimum overlap of short birdcalls
MIN_OVERLAP = 0.5
# the min duration of a 'long' bircall
LONG_BIRDCALL_LENGTH = 5 # seconds
# how long should we at least hear this birdcall in a segment
LONG_OVERLAP_SECONDS = 2 # seconds


class DataGeneratorTestset(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, batch_size=32):
        'Initialization'
        self.batch_size = batch_size

        data_root = data_reading.test_data_base_dir + "example_test_audio/"
        label_root = data_reading.test_data_base_dir + "example_test_audio_metadata.csv"
        self.labels = pd.read_csv(label_root)
        self.files = glob.glob(f"{data_root}/*")

        self.X = []
        self.y = []

        self.__data_generation()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        return self.X[index:index+self.batch_size], self.y[index:index+self.batch_size]

    def __data_generation(self):

        for file in self.files:
            fragments = preprocess(file, feature_extractor)

            for i, fragment in enumerate(fragments):
                t_start, t_end = i * 5, i * 5 + 5
                self.X.append(fragment)

                detected_birds = set()
                for row in self.labels.itertuples():
                    if not row.file_id in file:
                        continue

                    overlap = compute_overlap(t_start, t_end, row.time_start, row.time_end)
                    if overlap <= 0:
                        continue

                    label_duration = row.time_start - row.time_end
                    if label_duration < LONG_BIRDCALL_LENGTH and overlap / label_duration < MIN_OVERLAP:
                        # Short birdcall
                        detected_birds.add(bird_code.get(row.ebird_code, None))
                    elif overlap > LONG_OVERLAP_SECONDS:
                        # Long birdcall
                        detected_birds.add(bird_code.get(row.ebird_code, None))


                y = np.array([1 if i in detected_birds else 0 for i in bird_code.values()])
                self.y.append(y)

        self.X = np.concatenate(self.X)  # concat from (32, 1, 16, 7, 2048) to (32, 16, 7, 2048)
        self.y = np.array(self.y)

if __name__ == '__main__':
    print("\nTRAIN\n")
    d = DataGenerator("preprocessed")
    X, y = d[0]
    print(X.shape, y.shape)

    print("\nTEST\n")
    d = DataGeneratorTestset()
    X, y = d[0]
    print(X.shape, y.shape)