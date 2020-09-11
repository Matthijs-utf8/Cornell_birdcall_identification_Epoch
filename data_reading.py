import librosa
import pandas
import numpy as np
import sounddevice
import os

# Read the config file or create it if it doesn't exist yet
def read_config():
    if not os.path.isfile('.config'):
        print("Where is the kaggle competition folder located?")
        location = input()
        with open(".config", "w") as config_file:
            config_file.write("data_folder=" + location)
            return location + "/"
    else:
        with open(".config") as config_file:
            config_lines = config_file.readlines()
            split = (line.split("=", 1) for line in config_lines)
            return next(value for key, value in split if key == "data_folder") + "/"

test_data_base_dir = read_config()

# Read train.csv from the kaggle directory into a pandas dataframe
def get_train_metadata():
    return pandas.read_csv(test_data_base_dir + "train.csv")

def get_validation_metadata():
    return pandas.read_csv(test_data_base_dir + "example_test_audio_metadata.csv")

def get_test_example_files():
    directory = test_data_base_dir + "example_test_audio/"

    return [
        directory + file
        for file in os.listdir(directory)
    ]
