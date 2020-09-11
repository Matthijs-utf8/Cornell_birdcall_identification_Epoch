# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:10:54 2020

@author: siets
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import Noise_Extractor
import sklearn
import pydub
import data_reading
import utils
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
import dataloader
import warnings

warnings.filterwarnings("ignore")

universal_sample_rate = 32000

melspectrogram_parameters = {
    "n_mels": 128,
    "fmin": 20,
    "fmax": 16000
}

# Add the path of each file to the train.csv
base_dir = data_reading.read_config()
df_train = pd.read_csv(base_dir + "train.csv")

#Method copied from notebook Xavier used to do our 0.560 submission
def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V
    

#Method to read audio from a specified path and return the mel-spectrograms
def make_spectrograms(batch, sr=32000):
    
    #Specify resulting list of images
    result = []
    
    #Loop over audio in 5 second windows
    for samples in batch:
        if len(samples) != 5 * sr:
            raise ValueError("Samples have wrong length; should be 160000 with sr=32000")
        
        """
        Block of code copied from another notebook to convert an audio window of five seconds into a mel spectrogram
        """
        y = samples.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * 224 / height), 224))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        #Append image to result. Image has shape (3, 224, 547). np.array(result) has shape (n, 3, 224, 547)
        result.append(image)
    
    #Return the array of spectrograms
    #Lose the first one since it only contains zeros
    
    # if np.array(result).shape != (32, 3, 224, 547):
    #     raise ValueError("Dit gaat weer helemaal fout")
    
    return np.array(result)


if __name__ == "__main__":
    model_path = "C:/Users/siets/OneDrive/Documenten/Sietse/Team Epoch/best_keras.pth.h5"
    model = keras.models.load_model(model_path, custom_objects={
    'recall_m': 0,
    'precision_m': 0,
    'f1_m': 0
    })
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=[keras.metrics.CategoricalAccuracy(), utils.f1_m, utils.precision_m, utils.recall_m])
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                                  patience=5, cooldown=2, min_lr=1e-9)


    
    
    
    gen = dataloader.DataGeneratorHDF5(base_dir + "test_traindataset.hdf5")
    
    for batch in gen:
        spectrogram_batch = make_spectrograms(batch[0], sr=32000)
        
        print(type(batch[1]))
        
        model.fit((spectrogram_batch, batch[1]), callbacks=[reduce_lr],
              epochs=1, workers=1)
       # model.fit((spectrogram_batch, batch[1]), epochs=1)
