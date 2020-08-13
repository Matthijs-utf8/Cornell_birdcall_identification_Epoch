import argparse
import numpy as np

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras import layers
import dataloader
from birdcodes import bird_code

class LRTensorBoard(TensorBoard):
    "Tensorboard that also logs learning rate"
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=512, type=int, help="Training batch size")
    parser.add_argument("--workers", default=4, type=int, help="Number of dataloader workers")
    parser.add_argument("--feature_mode", default="spectrogram", type=str, help="Possible values: 'spectrogram' or 'resnet'")

    args = parser.parse_args()

    np.random.seed(args.seed)
    use_resnet = args.feature_mode == "resnet"

    spectrogram_dim = (250, 257)

    
    # input_shape = (16, 7, 2048)
    input_shape = (8, 9, 2048)
    if not use_resnet:
        input_shape = spectrogram_dim + (1,)


    if not use_resnet:
        data_generator = dataloader.DataGenerator("spectrograms", batch_size=args.batch_size, dim=input_shape)
    else:
        data_generator = dataloader.DataGenerator("preprocessed2", batch_size=args.batch_size, dim=input_shape)
    
    print("len =", len(bird_code))

    if not use_resnet:
        model = keras.models.Sequential([
            layers.Conv2D(16, (5, 5), activation='relu', input_shape=input_shape),
            layers.MaxPool2D(),
            layers.Conv2D(16, (5, 5), activation='relu'),
            layers.MaxPool2D(),
            layers.Conv2D(16, (5, 5), activation='relu'),
            layers.Flatten(),
            layers.Dense(len(bird_code), activation="sigmoid"),
        ])
    else:
        model = kears = keras.models.Sequential([
            layers.GlobalMaxPool2D(input_shape=input_shape),
            layers.Dense(1024),
            layers.Dense(len(bird_code)),
        ])

    print("trainable count:", len(model.trainable_variables))
    optimizer = keras.optimizers.Adam(
        learning_rate=args.lr,
        # decay=1e-2,
    )

    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=[keras.metrics.CategoricalAccuracy(), f1_m, precision_m, recall_m])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=5, min_lr=1e-9)

    tensorboard_callback = LRTensorBoard(log_dir="logs")

    model.fit(data_generator, callbacks=[reduce_lr, tensorboard_callback], epochs=args.epochs, workers=args.workers)
    model.save("models/baseline")

    model = keras.models.load_model("models/baseline",
                                    custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})

    test_generator = dataloader.DataGeneratorTestset()
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_generator)
    print("EVALUATION:")
    print("loss      ", loss)
    print("accuracy  ", accuracy)
    print("f1_score  ", f1_score)
    print("precision ", precision)
    print("recall    ", recall)
