import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras

import dataloader
import models
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=512, type=int, help="Training batch size")
    parser.add_argument("--workers", default=1, type=int, help="Number of dataloader workers, may work incorrectly")
    parser.add_argument("--feature-mode", default="spectrogram", type=str,
                        help="Possible values: 'spectrogram' or 'resnet' for preprocessed by resnet base")
    parser.add_argument("--arch", default="cnn", type=str,
                        help="Network architecture, possible values: 'cnn', 'resnet-head', or '1d-conv' or 'resnet-full")
    parser.add_argument("--name", type=str, help="The experiment run name for tensorboard")

    args = parser.parse_args()

    np.random.seed(args.seed)

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices for multigpu strategy: {strategy.num_replicas_in_sync}')

    with strategy.scope():
        if args.arch == "cnn":
            model, input_shape, channels = models.CNN()
        elif args.arch == "resnet-head":
            model, input_shape, channels = models.ResNetHead()
        elif args.arch == "1d-conv":
            model, input_shape, channels = models.Conv1D()
        elif args.arch == "resnet-full":
            model, input_shape, channels = models.ResNet()
        else:
            raise NotImplementedError("Model type not supported")

        # print("trainable count:", len(model.trainable_variables))
        optimizer = keras.optimizers.Adam(
            learning_rate=args.lr,
            # decay=1e-2,
        )

        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                      metrics=[keras.metrics.CategoricalAccuracy(), utils.f1_m, utils.precision_m, utils.recall_m])


    # Data
    data_generator = dataloader.DataGenerator("spectrograms", batch_size=args.batch_size, dim=input_shape,
                                              channels=channels)
    data_generator, data_generator_val = data_generator.split(0.1)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                                  patience=5, cooldown=2, min_lr=1e-9)

    tensorboard_callback = utils.LRTensorBoard(log_dir=f"logs/{args.name}", settings_to_log=str(args))

    save_best_callback = keras.callbacks.ModelCheckpoint(filepath="models/" + args.name + ".val_f1.{val_f1_m:.3f}.h5",
                                                         save_best_only=True, monitor='val_f1_m')
    # callback = keras.callbacks.EarlyStopping(monitor='val_f1_m', patience=5)

    model.fit(data_generator, callbacks=[reduce_lr, tensorboard_callback, save_best_callback],
              epochs=args.epochs, workers=args.workers, validation_data=data_generator_val)
    model.save("models/" + args.name + ".h5")
