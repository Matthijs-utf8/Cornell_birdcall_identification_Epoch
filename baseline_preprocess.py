import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

from tqdm import tqdm
import numpy as np
from scipy.signal import resample
import librosa
from matplotlib import pyplot as plt

from Noise_Extractor import filter_sound, get_frames
import data_reading

window_size = 220
universal_sample_rate = 22000
spectrogram_slices_per_input = universal_sample_rate * 5 // window_size # = 5 seconds

def preprocess(file_path, feature_extractor: keras.models.Model):
    sound, sample_rate = librosa.load(file_path)

    if sample_rate != universal_sample_rate:
        sound = resample(sound, int(universal_sample_rate * (len(sound) / sample_rate)))

    # print(sound.shape)

    spectrogram = np.array([
        np.fft.fft(sound[i * window_size : (i + 1) * window_size]).real
        for i in range(0, sound.shape[0] // window_size)
    ])

    repeated = np.repeat(
        spectrogram.reshape((1, spectrogram.shape[0], window_size, 1)),
        3,
        axis=3
    )

    return [
        feature_extractor.predict(repeated[:, i * spectrogram_slices_per_input : (i + 1) * spectrogram_slices_per_input])

        for i in range(repeated.shape[1] // spectrogram_slices_per_input)
    ]

def tf_fourier(file_path):
    sound, sample_rate = librosa.load(file_path)

    if sample_rate != universal_sample_rate:
        sound = resample(sound, int(universal_sample_rate * (len(sound) / sample_rate)))

    print(sound.shape)

    spectrogram = np.array([
        tf.math.real(
            tf.signal.fft(
                sound[i * window_size : (i + 1) * window_size]
            )
        )
        for i in range(0, sound.shape[0] // window_size)
    ])

    spectrogram = np.array([
        np.fft.fft(sound[i * window_size : (i + 1) * window_size]).real
        for i in range(0, sound.shape[0] // window_size)
    ])

    plt.imshow(spectrogram)
    plt.show()

    


feature_extractor: keras.models.Model = ResNet50(input_shape=(spectrogram_slices_per_input, window_size, 3), include_top=False)

if __name__ == "__main__":
    import sys
    import os

    output_folder = "preprocessed"
    
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for birdcode in tqdm(sys.argv[1:]):
        fragment_id = 0

        path = data_reading.test_data_base_dir + "train_audio/" + birdcode + "/"
        for file_name in os.listdir(path):
            fragments = tf_fourier(path + file_name)
            exit()

            for fragment in fragments:
                np.save("preprocessed/" + birdcode + "_" + str(fragment_id), fragment)
                fragment_id += 1


