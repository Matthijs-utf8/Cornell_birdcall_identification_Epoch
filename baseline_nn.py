from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras import layers
import dataloader
from birdcodes import bird_code

input_shape = (16, 7, 2048)



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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == "__main__":
    data_generator = dataloader.DataGenerator("preprocessed", batch_size=512)
    print("len =", len(bird_code))

    model = keras.models.Sequential([
        # keras.Input(input_shape), # shape=(16, 9, 2048)
        layers.Conv2D(1024, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(len(bird_code), activation="sigmoid"),
    ])

    print("trainable count:", len(model.trainable_variables))

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[keras.metrics.CategoricalAccuracy(), f1_m,precision_m, recall_m])

    model.fit(data_generator, epochs=5)
    model.save("baseline.tf")

    model = keras.models.load_model("baseline.tf")

    test_generator = dataloader.DataGeneratorTestset()
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_generator)
    print("EVALUATION:")
    print("loss      ", loss)
    print("accuracy  ", accuracy)
    print("f1_score  ", f1_score)
    print("precision ", precision)
    print("recall    ", recall)