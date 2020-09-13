import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from dataloader import DataGeneratorHDF5
import models
import utils
import warnings
import data_reading
import pandas as pd
import time
import tqdm
import train_on_melspectrograms as tom

base_dir = data_reading.read_config()
df_train = pd.read_csv(base_dir + "train.csv")

if __name__ == "__main__":
    t = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=1, type=int, help="Training batch size")
    parser.add_argument("--workers", default=1, type=int, help="Number of dataloader workers, may work incorrectly")
    parser.add_argument("--feature-mode", default="spectrogram", type=str,
                        help="Possible values: 'spectrogram' or 'resnet' for preprocessed by resnet base")
    parser.add_argument("--arch", default="resnet-full", type=str,
                        help="Network architecture, possible values: 'cnn', 'resnet-head', or '1d-conv' or 'resnet-full")
    parser.add_argument("--name", type=str, help="The experiment run name for tensorboard")

    args = parser.parse_args()

    np.random.seed(args.seed)

    # strategy = tf.distribute.MirroredStrategy()
    # print(f'Number of devices for multigpu strategy: {strategy.num_replicas_in_sync}')



    if args.arch == "cnn":
        model, input_shape, channels = models.CNN()
    elif args.arch == "resnet-head":
        model, input_shape, channels = models.ResNetHead()
    elif args.arch == "1d-conv":
        model, input_shape, channels = models.Conv1D()
    elif args.arch == "resnet-full":
        model, input_shape, channels = models.ResNet()
    elif args.arch == "saved_model":
        model, input_shape, channels = models.savedModel()
    else:
        raise NotImplementedError("Model type not supported")
        
        


    # # print("trainable count:", len(model.trainable_variables))
    # optimizer = keras.optimizers.Adam(
    #     learning_rate=args.lr
    #     # decay=1e-2,
    # )
    

    # model.compile(loss="binary_crossentropy", optimizer=optimizer,
    #               metrics=[keras.metrics.CategoricalAccuracy(), utils.f1_m, utils.precision_m, utils.recall_m])

    # model.summary()
    
    # for x in range(1000):
    #     if type(model.layers[x]) == type(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))):
    #         print("Hoeezzeeee")
    
    # for layer in model.layers:
    #     if type(layer) == type(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))):
    #         layer.data_format = 'channels_first'
    
    
    
    spectrograms = tom.read_audio(df_train['full_path'][0])
    spect = np.reshape(spectrograms[0], (1,) + spectrograms[0].shape)
    print(spect.shape)
    
    print(model.predict(spect))
    
    
    
    
    
    
    with DataGeneratorHDF5("D:/Sietse/Datasets/test_frequency.hdf5") as ds:
        X, y = ds[0]
        
        X = np.reshape(X[0], (1,) + X[0].shape)
        y = np.reshape(y[0], (1,) + y[0].shape)
        print(X.shape)
        print(y.shape)

        
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
        #                                               patience=5, cooldown=2, min_lr=1e-9)
        #tensorboard_callback = utils.LRTensorBoard(log_dir=f"logs/{args.name}", settings_to_log=str(args))
        
    
        # save_best_callback = keras.callbacks.ModelCheckpoint(filepath="models/" + args.name + ".val_f1.{val_f1_m:.3f}.h5",
        #                                                      save_best_only=True, monitor='val_f1_m')
        # callback = keras.callbacks.EarlyStopping(monitor='val_f1_m', patience=5)
        
        print("Started fitting, time elapsed so far:", time.perf_counter() - t, "seconds")

        model.predict(X)
        
        model.fit(X, y, batch_size=1, epochs=args.epochs)
        model.save("models/" + args.name + ".h5")
