import numpy as np
import pandas as pd
import librosa
import random
import warnings
import sounddevice as sd
import data_reading
import birdcodes
import scipy
import Noise_Extractor
import sound_shuffling


# A function that prevents warnings when loading in files with librosa
warnings.simplefilter("ignore")

# Add the path of each file to the train.csv
base_dir = data_reading.read_config()
df_train = pd.read_csv(base_dir + "train.csv")

####### !!!!!!!!!!!!!!! ##########
####### Run these two lines below once if you've never run this file before. It adds a filepath to each file in train.csv #########
# df_train['full_path'] = base_dir + "train_audio/" + df_train['ebird_code'] + '/' + df_train['filename']
# df_train.to_csv(base_dir + "train.csv")

def resample(samples, sampling_rate, universal_sr):
	
	if sampling_rate == universal_sr: return samples
	
	else: return scipy.signal.resample(x=samples, num=int( universal_sr * (len(samples)/sampling_rate) ) )

def extract_noise(samples, sampling_rate, window_width=2048, stepsize=512, verbose=False):
	
	return Noise_Extractor.filter_sound(samples, sampling_rate, window_width=window_width, stepsize=stepsize, verbose=verbose)

def create_shuffled_dataset(dataset_size=3, metrics=["country"], files_to_combine=2, universal_sr=22050, clip_seconds=5):
	
	for _ in range(dataset_size):
		
		
		new_dataframe = sound_shuffling.filter_metadata_by_metrics(df_train, metrics=["country", "species"], nr_of_files=files_to_combine)
		
		
		random_files = sound_shuffling.pick_files_at_random(new_dataframe, nr_of_files=files_to_combine)
		
		
		combined_file, labels = sound_shuffling.combine_files(files=random_files, universal_sr=universal_sr, seconds=clip_seconds)


if __name__ == "__main__":
	
	create_shuffled_dataset(dataset_size=3, metrics=["country"], files_to_combine=2, universal_sr=22050, clip_seconds=5)