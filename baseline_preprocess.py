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

window_size = 440
universal_sample_rate = 22000
spectrogram_slices_per_input = universal_sample_rate * 5 // window_size # = 5 seconds

def preprocess(file_path, feature_extractor: keras.models.Model):
    sound, sample_rate = librosa.load(file_path)

    if sample_rate != universal_sample_rate:
        sound = resample(sound, int(universal_sample_rate * (len(sound) / sample_rate)))

    # print(sound.shape)

    spectrogram = tf.abs(
        tf.signal.stft(sound, window_size, window_size)
    ).numpy()

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
        pass

    spectrogram = tf.abs(
        tf.signal.stft(tf.reshape(sound, [1, -1]), window_size, window_size, fft_length=512)
    )[0]

    slices = []
    for i in range(spectrogram.shape[0] // spectrogram_slices_per_input):
        spectrogram_slice = spectrogram[i * spectrogram_slices_per_input : (i + 1) * spectrogram_slices_per_input]
        slices.append(spectrogram_slice)
        
        # print(spectrogram_slice.shape)
        # plt.imshow(spectrogram_slice)
        # plt.show()

    return slices

    


feature_extractor: keras.models.Model = ResNet50(input_shape=(spectrogram_slices_per_input, window_size, 3), include_top=False)

if __name__ == "__main__":
    import sys
    import os

    output_folder = "spectrograms"
    
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for birdcode in sys.argv[1:]:
        print(birdcode)
        fragment_id = 0

        path = data_reading.test_data_base_dir + "train_audio/" + birdcode + "/"
        for file_name in tqdm(os.listdir(path)):
            fragments = tf_fourier(path + file_name)

            for fragment in fragments:
                np.save(output_folder + "/" + birdcode + "_" + str(fragment_id), fragment)
                fragment_id += 1


