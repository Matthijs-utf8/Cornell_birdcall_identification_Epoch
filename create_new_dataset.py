import numpy as np
import pandas as pd
import os
import librosa
import random
import warnings
import scipy
import re
import pickle
import sounddevice as sd

import birdcodes
import data_reading
import Noise_Extractor
import sound_shuffling
import preprocessing

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

df_train['full_path'] = base_dir + "train_audio/" + df_train['ebird_code'] + '/' + df_train['filename']
df_train.to_csv(base_dir + "train.csv")

""" A function that uses a few methods from sound_shuffling.py to be able to easily create a new shuffled dataset. """
def create_shuffled_dataset(nr_of_files, files_to_combine, metrics=[], clip_seconds=5):
	"""
	:param nr_of_files: int, number of 5 second files that should be created
	:param metrics: list (or empty list), common metrics that the birdsounds should have
	:param files_to_combine: int, number of birds to overlap eachother
	:param clip_seconds: int, number of seconds that the new clip should be
	"""
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
		filename += ".mp3"

		# Add the files to the list
		all_files.append(filename)

		# Combine the files. Normalization happens in this step as well.
		combined_file, labels = sound_shuffling.combine_files(files=random_files,
															  universal_sr=22050,
															  seconds=clip_seconds)

		# Save the files as .mp3 files
		preprocessing.write(f=save_dir + "/" + filename, x=combined_file, sr=universal_sr)

		# Add labels to the list
		all_labels.append(labels)

	# Save the labeled files in a dictionary as a pickle
	labeled_data = dict(zip(all_files, all_labels))
	f = open(save_dir + "/labels.pkl","wb")
	pickle.dump(labeled_data, f)
	f.close()


def create_denoised_dataset(nr_of_files):
	"""
	Create a dataset that is made up of original audio, but denoised, resampled to 22050Hz and normalized.
	Processes: Original audio -> Resample -> Denoise -> Normalize -> Denoised audio
	:param nr_of_files: int, number of 5 second files that should be created
	"""
	all_labels = []
	all_files = []

	# Create a directory to save  the new files in
	save_dir = base_dir + "/train_audio_denoised"
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	for _ in range(nr_of_files):

		# Pick a random file from the training metadata
		random_file = sound_shuffling.pick_files_at_random(df_train,
															nr_of_files=1)

		# Create a new filename using regular expressions
		filename = re.findall("[XC]{2,}[0-9]{2,}", random_file["filename"].tolist()[0])[0] + "_denoised.mp3"

		# Add the files to the list
		all_files.append(filename)

		# Read in the samples
		samples, sampling_rate = librosa.load(random_file["full_path"].tolist()[0])

		# Resample the audio to 22050 Hertz
		samples = preprocessing.resample(samples,
										 sampling_rate,
										 universal_sr=22050)

		# Denoise the audio. Normalization happens in this step as well.
		denoised_samples = preprocessing.extract_noise(samples,
													   sampling_rate,
													   window_width=2048,
													   step_size=512,
													   verbose=False)

		# Save the files as .mp3 files
		preprocessing.write(f=save_dir + "/" + filename, x=denoised_samples, sr=universal_sr)

		# Add labels to the list
		all_labels.append(birdcodes.bird_code.get(random_file["ebird_code"].tolist()[0]))

	# Save the labeled files in a dictionary as a pickle
	labeled_data = dict(zip(all_files, all_labels))
	f = open(save_dir + "/labels.pkl", "wb")
	pickle.dump(labeled_data, f)
	f.close()


"""Method to create a dataset with different levels of noise"""
def random_noise_dataset():

	# Create folder to store dataset if this folder does not exist yet
	if not os.path.exists(base_dir + "train_audio_random_noise"):
		os.mkdir(base_dir + "train_audio_random_noise")

	# Create folders to store audio from different birds
	for code in birdcodes.bird_code.keys():
		if not os.path.exists(base_dir + "train_audio_random_noise/" + code):
			os.mkdir(base_dir + "train_audio_random_noise/" + code)

	# Add a full path to dataset in df_train
	df_train['full_path_random_noise'] = base_dir + "train_audio_random_noise/" + df_train[
		'ebird_code'] + '/random_noise_' + df_train['filename']

	# Add noise for all audio files in full path
	for i in range(len(df_train['full_path'])):
		# Load file
		samples, sr = librosa.load(df_train['full_path'][i])

		# Add noise to samples from a standard normal distribution
		samples = sound_shuffling.add_white_noise(samples, target_snr=np.random.normal(4.5, 2.0))

		# Save samples
		preprocessing.write(base_dir + "train_audio_random_noise/" + df_train['ebird_code'][i] + '/random_noise_' +
							df_train['filename'][i], 22050, samples)


"""Method to create dataset with background noise from other files"""
def random_background_dataset():

	# Create folder to store dataset if this folder does not exist yet
	if not os.path.exists(base_dir + "train_audio_random_background"):
		os.mkdir(base_dir + "train_audio_random_background")

	# Create folders for different birds
	for code in birdcodes.bird_code.keys():
		if not os.path.exists(base_dir + "train_audio_random_background/" + code):
			os.mkdir(base_dir + "train_audio_random_background/" + code)

	# Adjust df_train to add path to dataset
	df_train['full_path_random_background'] = base_dir + "train_audio_random_background/" + df_train[
		'ebird_code'] + '/random_background_' + df_train['filename']

	# Add background noises for all files in full path
	for i in range(len(df_train['full_path'])):
		# Load file
		samples, sr = librosa.load(df_train['full_path'][i])

		# Add background noise
		samples = sound_shuffling.add_random_background_noise(samples, sr)

		# Save file
		preprocessing.write(
			base_dir + "train_audio_random_background/" + df_train['ebird_code'][i] + '/random_background_' +
			df_train['filename'][i], 22050, samples)



""" A function that creates additional shifted data to append to original dataset.
	"Shift" can be Amplitude, Frequency or Time.
	"nr_files" is how much shifted files will be added"""
def create_shifted_dataset(shift, universal_sr, clip_seconds, nr_files):
	all_labels = []
	all_files = []

	# Create a directory to save  the new files in
	save_dir = base_dir + "/train_audio_" + str(shift) + "_shifted"
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	for _ in range(nr_files):

		# Get the sorted dataframe and pick ONE random file from it
		new_dataframe = sound_shuffling.filter_metadata_by_metrics(df_train, metrics=[], nr_of_files=1)
		random_file = sound_shuffling.pick_files_at_random(new_dataframe, nr_of_files=1)

		# i = list(random_file["Unnamed: 0"])[0]
		random_sample, sr = librosa.load(random_file["full_path"].loc[0])
		# print("RANDOM sample \n", random_sample, "\n")

		if shift == "Amplitude":
			# Random shift
			# n_steps = float(random.randrange(0, 1500))/100 # lower volume is between 0 and 1, so no negative numbers
			n_steps = random.randint(0, 15)
			shifted_file = sound_shuffling.amplitude_shift(random_sample, n_steps)

		elif shift == "Frequency":
			# Random shift
			n_steps = random.randint(-15, 15)

			shifted_file = sound_shuffling.frequency_shift(random_sample, universal_sr, n_steps)

		elif shift == "Time":
			# Random shift
			n_steps = random.randint(-15, 15)

			shifted_file = sound_shuffling.time_stretch(random_sample, n_steps)

		else:
			print("Wrong type of shift, you can use: Amplitude, Frequency or Time.")
			exit()

		print("SHIFTED sample \n", shifted_file, "\n")

		# Create a new filename
		filename = random_file["filename"] + "_" + str(shift) + "_shifted"

		# Save the files as .mp3 files
		preprocessing.write(f=save_dir + "/" + filename + ".mp3", sr=universal_sr, x=shifted_file)

		# Add the file and label to the lists
		all_files.append(filename + ".mp3")
		all_labels.append(labels)

	# Save the labeled files in a dictionary as a pickle
	labeled_data = dict(zip(all_files, all_labels))
	f = open(save_dir + "/dict.pkl","wb")
	pickle.dump(labeled_data, f)
	f.close()

	return

if __name__ == "__main__":

	# Create a dictionary for storing the labels that accompany the files

	""" Hyperparameters """
	dataset_size = 10
	metrics = [] # If list is empty, it chooses from all the files
	files_to_combine = 2 # Number of files to merge each time
	universal_sr = 22050
	clip_seconds = 5 # The length of the new clips in seconds

	# create_denoised_dataset(nr_of_files=dataset_size)

	# create_shuffled_dataset(
	# 					  nr_of_files=dataset_size,
	# 					  metrics=metrics,
	# 					  files_to_combine=2,
	# 					  clip_seconds=clip_seconds
	# 					  )

	# create_shuffled_dataset(
	# 					  nr_of_files=dataset_size,
	# 					  metrics=metrics,
	# 					  files_to_combine=4,
	# 					  clip_seconds=clip_seconds
	# 					  )

	# create_shuffled_dataset(
	# 					  nr_of_files=dataset_size,
	# 					  metrics=metrics,
	# 					  files_to_combine=6,
	# 					  clip_seconds=clip_seconds
	# 					  )
