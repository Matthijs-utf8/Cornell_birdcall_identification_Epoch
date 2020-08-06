from tensorflow import keras
from tensorflow.keras import layers
import dataloader
from birdcodes import bird_code

input_shape = (16, 7, 2048)

if __name__ == "__main__":
    data_generator = dataloader.DataGenerator("preprocessed")
    print("len =", len(bird_code))

    model = keras.models.Sequential([
        # keras.Input(input_shape), # shape=(16, 9, 2048)
        layers.Conv2D(1024, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(len(bird_code), activation="sigmoid"),
    ])

    print("trainable count:", len(model.trainable_variables))

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[keras.metrics.CategoricalAccuracy()])

    model.fit(data_generator, epochs=5)
    model.save("baseline.tf")

    model = keras.models.load_model("baseline.tf")

    test_generator = dataloader.DataGeneratorTestset()
    score = model.evaluate(test_generator)
    print("METRIC:")
    print(score)