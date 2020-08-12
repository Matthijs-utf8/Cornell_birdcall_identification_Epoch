import argparse
import numpy as np

from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras import layers
import dataloader
from birdcodes import bird_code



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

    args = parser.parse_args()

    np.random.seed(args.seed)

    spectrogram_dim = (250, 257)

    input_shape = (16, 7, 2048)
    # input_shape = spectrogram_dim + (1,)


    # data_generator = dataloader.DataGenerator("spectrograms", batch_size=args.batch_size, dim=input_shape)
    data_generator = dataloader.DataGenerator("preprocessed2", batch_size=args.batch_size, dim=input_shape)
    print("len =", len(bird_code))

    model = keras.models.Sequential([
        # keras.Input(input_shape), # shape=(16, 9, 2048)
        layers.Conv2D(16, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPool2D(),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.Flatten(),
        layers.Dense(len(bird_code), activation="sigmoid"),
    ])

    print("trainable count:", len(model.trainable_variables))
    optimizer = keras.optimizers.Adam(
        learning_rate=args.lr,
    )

    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=[keras.metrics.CategoricalAccuracy(), f1_m, precision_m, recall_m])

    model.fit(data_generator, epochs=args.epochs, workers=args.workers)
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
