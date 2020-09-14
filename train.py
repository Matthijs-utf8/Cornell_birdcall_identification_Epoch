import argparse

import numpy as np
from tensorflow import keras
from dataloader import DataGeneratorHDF5
import models
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=2, type=int, help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
   
 
    
    
    parser.add_argument("--name", type=str, help="The experiment run name for tensorboard")
    parser.add_argument("--data_path", default="", type=str, help="Path to the dataset")
    parser.add_argument("--model_path", default="", type=str, help="Path to model")
    args = parser.parse_args()

    np.random.seed(args.seed)

    model_path = args.model_path if args.model_path != "" else "C:/Users/siets/OneDrive/Documenten/Sietse/Team Epoch/best_keras.pth.h5"
    
    model, input_shape, channels = models.savedModel(model_path)

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=[keras.metrics.CategoricalAccuracy(), utils.f1_m, utils.precision_m, utils.recall_m])

    data_path = args.data_path if args.data_path != "" else "D:/Sietse/Datasets/test_frequency.hdf5"
    with DataGeneratorHDF5(data_path) as ds:
        
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
        #                                               patience=5, cooldown=2, min_lr=1e-9)
        #tensorboard_callback = utils.LRTensorBoard(log_dir=f"logs/{args.name}", settings_to_log=str(args))
        
    
        # save_best_callback = keras.callbacks.ModelCheckpoint(filepath="models/" + args.name + ".val_f1.{val_f1_m:.3f}.h5",
        #                                                      save_best_only=True, monitor='val_f1_m')
        # callback = keras.callbacks.EarlyStopping(monitor='val_f1_m', patience=5)
        for X, y in ds:

            print('X shape \n', X.shape)
            # print('X array\n', X)
            y = np.stack([y for _ in range(3)])
            # y = np.array([y,y,y])
            print('y shape \n', y.shape)
            # print('y array \n', y)
            data = (X, y)
            model.fit(data, batch_size=args.batch_size, epochs=args.epochs)
            model.save("models/" + args.name + ".h5")
