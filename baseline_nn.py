from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from Noise_Extractor import filter_sound, get_frames
import data_reading
import numpy as np

window_size = 2048

def preprocess(frames, feature_extractor: keras.models.Model):
    filtered, sr = filter_sound(frames)
    print(sr)

    print(filtered.shape)

    windows = get_frames(filtered, window_size, window_size)

    spectrogram = np.array([
        np.fft.fft(filtered[i * window_size : (i + 1) * window_size])
        for i in range(0, filtered.shape[0] // window_size)
    ])



    print(spectrogram.shape)

    repeated = np.array(np.repeat(
        spectrogram.reshape((1, spectrogram.shape[0], window_size, 1)),
        3,
        axis=3
    ))

    return [
        feature_extractor.predict(repeated[:, i * window_size : (i + 1) * window_size])

        for i in range(spectrogram.shape[0] // window_size)
    ]

feature_extractor: keras.models.Model = ResNet50(input_shape=(window_size, window_size, 3), include_top=False)

preprocessed = [
    preprocess(file_path, feature_extractor) 
    for file_path in data_reading.get_test_example_files()
]

