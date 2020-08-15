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
import argparse

window_size = 440
universal_sample_rate = 22000
spectrogram_slices_per_input = universal_sample_rate * 5 // window_size # = 5 seconds

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

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
    sound, sample_rate = librosa.load(file_path)

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

    return slices

spectrogram_shape = (250, 257)

resnet: keras.models.Model = ResNet50(input_shape=(spectrogram_shape + (3,)), include_top=False)

if __name__ == "__main__":
    import sys
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_mode", default="spectrogram", type=str, help="Possible values: 'spectrogram' or 'resnet'")
    parser.add_argument("--dir", default="preprocessed2", type=str, help="Where to place the preprocessed files")
    parser.add_argument("-b", "--bird_codes", nargs="+", type=str, help="List of birdcodes indicating which files need to be processed")
    args = parser.parse_args()
    
    output_dir = args.dir
    use_resnet = args.feature_mode == "resnet"

    print(args.bird_codes)    
    
    # Create output_dir if it doesn't exist yet
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Process all files based on the birdcodes in the arguments
    for birdcode in args.bird_codes:
        print(birdcode)

        fragment_id = 0   # A unique identifier for each slice of 5 seconds
        path_to_birdsound_dir = data_reading.test_data_base_dir + "train_audio/" + birdcode + "/"

        for file_name in tqdm(os.listdir(path_to_birdsound_dir)):
            if use_resnet:
                fragments = preprocess(path_to_birdsound_dir + file_name, resnet)
            else:
                fragments = tf_fourier(path_to_birdsound_dir + file_name)

            for fragment in fragments:
                np.savez_compressed(output_dir + "/" + birdcode + "_" + str(fragment_id), fragment)
                fragment_id += 1


