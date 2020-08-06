import glob
import numpy as np
from tensorflow import keras

from baseline_nn import bird_code


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
        y = np.empty((batch_size,), dtype=int)

        # Generate data
        for i, file in enumerate(files_temp):
            # Store sample

            X[i,] = np.load(file)

            # Store class
            bird_name = file.split("/")[-1].split("_")[0]
            y[i] = bird_code[bird_name]
        print("y", y)
        return X, keras.utils.to_categorical(y, num_classes=len(bird_code))

if __name__ == '__main__':
    d = DataGenerator("preprocessed")
    print(d[0])