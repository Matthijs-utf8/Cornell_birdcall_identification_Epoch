from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from Noise_Extractor import filter_sound, get_frames
import data_reading
import numpy as np
from scipy.signal import resample

window_size = 220
universal_sample_rate = 22000
spectrogram_slices_per_input = universal_sample_rate * 5 // window_size # = 5 seconds

def preprocess(frames, feature_extractor: keras.models.Model):
    filtered, sample_rate = filter_sound(frames)

    if sample_rate != universal_sample_rate:
        filtered = resample(filtered, int(universal_sample_rate * (len(filtered) / sample_rate)))

    print(filtered.shape)

    spectrogram = np.array([
        np.fft.fft(filtered[i * window_size : (i + 1) * window_size])
        for i in range(0, filtered.shape[0] // window_size)
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

feature_extractor: keras.models.Model = ResNet50(input_shape=(spectrogram_slices_per_input, window_size, 3), include_top=False)

preprocessed = [
    preprocess(file_path, feature_extractor) 
    for file_path in data_reading.get_test_example_files()[0:1]
]

