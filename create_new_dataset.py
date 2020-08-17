import numpy as np
import pandas as pd
import os
import librosa
import random
import warnings
import matplotlib.pyplot as plt
import data_reading
import scipy
import Noise_Extractor
import sound_shuffling
import sklearn
from PIL import Image
import preprocessing
import re
import pickle

### !!!!!!!!!! ###
# Comment out the lines below if you want different samples every time
random.seed(4)
np.random.seed(4)

# A function that prevents warnings when loading in files with librosa
warnings.simplefilter("ignore")

# Add the path of each file to the train.csv
base_dir = data_reading.read_config()
df_train = pd.read_csv(base_dir + "train.csv")

####### !!!!!!!!!!!!!!! ##########
####### Run these two lines below once if you've never run this file before. It adds a filepath to each file in train.csv #########
# df_train['full_path'] = base_dir + "train_audio/" + df_train['ebird_code'] + '/' + df_train['filename']
# df_train.to_csv(base_dir + "train.csv")

""" A function that uses a few methods from sound_shuffling.py to be able to easily create a new shuffled dataset. """
def create_shuffled_dataset(nr_of_files, metrics, files_to_combine, universal_sr, clip_seconds):

	all_labels = []
	all_files = []

	# Create a directory to save  the new files in
	save_dir = base_dir + "/train_audio_combined_" + str(files_to_combine) + "_" + str(metrics)
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	for _ in range(nr_of_files):

		# Get the sorted dataframe and pick random files from it (if metrics == None, the files will be randomly picked from the whole dataset)
		new_dataframe = sound_shuffling.filter_metadata_by_metrics(df_train, metrics=metrics, nr_of_files=files_to_combine)
		random_files = sound_shuffling.pick_files_at_random(new_dataframe, nr_of_files=files_to_combine)

		# Create a new filename using regular expressions
		filename = ""
		for file in random_files["filename"]:
			filename += "_" + re.findall("[XC]{2,}[0-9]{2,}", file)[0]

		# Add the files to the list
		all_files.append(filename + ".mp3")


		# Combine the files
		combined_file, labels = sound_shuffling.combine_files(files=random_files, universal_sr=universal_sr, seconds=clip_seconds)

		# Save the files as .mp3 files
		preprocessing.write(f=save_dir + "/" + filename + ".mp3", x=combined_file)

		# Add labels to the list
		all_labels.append(labels)

	# Save the labeled files in a dictionary as a pickle
	labeled_data = dict(zip(all_files, all_labels))
	f = open(save_dir + "/dict.pkl","wb")
	pickle.dump(labeled_data, f)
	f.close()

def random_noise_dataset():

	if not os.path.exists(base_dir + "train_audio_random_noise"):
		os.mkdir(base_dir + "train_audio_random_noise")

	for code in birdcodes.bird_code.keys():
		if not os.path.exists(base_dir + "train_audio_random_noise/" + code):
			os.mkdir(base_dir + "train_audio_random_noise/" + code)

	df_train['full_path_random_noise'] = base_dir + "train_audio_random_noise/" + df_train['ebird_code'] + '/random_noise_' + df_train['filename']

	for i in range(len(df_train['full_path'])):
		samples, sr = librosa.load(df_train['full_path'][i])
		samples = sound_shuffling.add_white_noise(samples, target_snr=np.random.normal(4.5, 2.0))

		preprocessing.write(base_dir + "train_audio_random_noise/" + df_train['ebird_code'][i] + '/random_noise_' + df_train['filename'][i], 22050, samples)

def random_background_dataset():

	if not os.path.exists(base_dir + "train_audio_random_background"):
		os.mkdir(base_dir + "train_audio_random_background")

	for code in birdcodes.bird_code.keys():
		if not os.path.exists(base_dir + "train_audio_random_background/" + code):
			os.mkdir(base_dir + "train_audio_random_background/" + code)

	df_train['full_path_random_background'] = base_dir + "train_audio_random_background/" + df_train['ebird_code'] + '/random_background_' + df_train['filename']

	for i in range(len(df_train['full_path'])):
		samples, sr = librosa.load(df_train['full_path'][i])
		samples = sound_shuffling.add_random_background_noise(samples, sr)

		preprocessing.write(base_dir + "train_audio_random_background/" + df_train['ebird_code'][i] + '/random_background_' + df_train['filename'][i], 22050, samples)

if __name__ == "__main__":

	# Create a dictionary for storing the labels that accompany the files

	""" Hyperparameters """
	dataset_size = 1
	metrics = [] # If list is empty, it chooses from all the files
	files_to_combine = 2 # Number of files to merge each time
	universal_sr = 22050
	clip_seconds = 5 # The length of the new clips in seconds
	window_width = 512 #

	create_shuffled_dataset(
						  nr_of_files=dataset_size,
						  metrics=metrics,
						  files_to_combine=files_to_combine,
						  universal_sr=universal_sr,
						  clip_seconds=clip_seconds
						  )

	"""Dont know if we want to use the denoised clips yet"""
# 		denoised_audio = extract_noise(
# 										combined_audio,
# 										universal_sr,
# 										window_width=2048,
# 										stepsize=512,
# 										verbose=False
# 										)
