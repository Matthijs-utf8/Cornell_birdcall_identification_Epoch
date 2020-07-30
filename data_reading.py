import librosa
import pandas
import numpy as np
import sounddevice
import os

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

def get_csv():
    return pandas.read_csv(test_data_base_dir + "train.csv")

def get_frames_from_index(index, csv):
    full_path = test_data_base_dir + "train_audio/" + csv['ebird_code'][index] + '/' + csv['filename'][index]
    return get_frames(full_path)

def default_test_frames():
    metadata = pandas.read_csv(test_data_base_dir + "train.csv")

    full_path = test_data_base_dir + "train_audio/" + metadata['ebird_code'][3] + '/' + metadata['filename'][3]

    return get_frames(full_path)


def get_frames(file_path):
    window_width = 2048
    stepsize = window_width

    data, sample_rate = librosa.load(file_path)
    nr_of_frames = (len(data) - window_width + stepsize) // stepsize
    
    frames = np.array([
        data[stepsize*n:stepsize*n+window_width]
        for n in range(nr_of_frames)
    ])

    return frames, sample_rate

if __name__ == "__main__":
    frames, sample_rate = default_test_frames()
    sounddevice.play(frames.flatten(), sample_rate)
    sounddevice.wait()
