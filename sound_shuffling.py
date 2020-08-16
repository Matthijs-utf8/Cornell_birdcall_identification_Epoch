import numpy as np
import pandas as pd
import librosa
import librosa.display as librosa_display
import random
import warnings
import sounddevice as sd
import data_reading
import birdcodes
import scipy
import matplotlib.pyplot as plt
import Noise_Extractor as ne

# A function that prevents warnings when loading in files with librosa
warnings.simplefilter("ignore")

# Add the path of each file to the train.csv
base_dir = data_reading.read_config()
df_train = pd.read_csv(base_dir + "train.csv")

####### !!!!!!!!!!!!!!! ##########
####### Run these two lines below once if you've never run this file before. It adds a filepath to each file in train.csv #########
# df_train['full_path'] = base_dir + "train_audio/" + df_train['ebird_code'] + '/' + df_train['filename']
# df_train.to_csv(base_dir + "train.csv")


""" Takes in the full df_train and outputs a dataframe with only 1 randomly choosen entry from a specified metric
EXAMPLE: let's say metrics=['country'] --> it chooses only files from the randomly choosen country of Potugal."""
def filter_metadata_by_metrics(dataframe, metrics=[], nr_of_files=2):

	# A list of all the possible metrics to sort by.
	possible_metrics = ['rating', 'playback_used', 'channels',
						'pitch', 'speed', 'species', 'number_of_notes',
						'bird_seen', 'sampling_rate', 'type',
						'volume', 'country', 'length']

	# Copy the original dataframe so we can work with it safely
	dataframe = dataframe

	# Pick files from the metadata randomly by the specified metrics
	for metric in metrics:

		# Create a while loop that only ends if we find a collection of metadata longer than the nr_of_files
		go_on = False
		loops = 0
		while go_on == False:

			loops += 1

			# Pick a random value from the metric (If it is 'country', pick from: Portugal, Canada, Sweden, etc...)
			random_pick = random.choice( list( df_train[metric] ) )

			# If the metadata has enough files to choose from
			if dataframe[dataframe[metric] == random_pick].shape[0] > nr_of_files:

				print("Randomly picked {}".format(random_pick) )

				# Redefine the dataframe to contain only the instances from the randomly picked metric
				dataframe = dataframe[dataframe[metric] == random_pick]

				go_on = True

			# If the loop goes on for too long, just skip the current metric
			if loops >= 1000:

				print("No combinations possible between {} and {}".format(metrics[0], metrics[1:]))

				go_on = True

	return dataframe


""" Takes in (any part of the) metadata and returns a random choice of n rows. """
def pick_files_at_random(dataframe, nr_of_files=2):

	# Check if there ae enough files available to choose from
	if dataframe.shape[0] < nr_of_files:
		raise ValueError("Can't choose {} files from {} samples.".format(nr_of_files, dataframe.shape[0]))

	# Pick random files from selection
	else:

		random_metadata = dataframe.sample(n=nr_of_files)
		print("Files picked:\n" + str(random_metadata[["ebird_code", "filename"]]))
		return random_metadata


""" Takes in one or more rows of the metadata and returns the corresponding soundfiles as numpy arrays. """
def combine_files(files, universal_sr=22050, seconds=5):

	# Get the labels that accompany the combined sounds
	labels = [birdcodes.bird_code.get(file) for file in files["ebird_code"]]

	# Initiate an array of zeros
	combined_sounds = np.zeros(universal_sr * seconds,)

	for file in files["full_path"]:

		# Load the samples
		samples, sampling_rate = librosa.load(file)

		# Resample the samples to a universal sampling rate
		if sampling_rate != universal_sr:

			# Resample with Scipy
			samples, sampling_rate = scipy.signal.resample(x=samples, num=int( universal_sr * (len(samples)/sampling_rate) ) ), universal_sr

			print("Sound has been resampled to {} Hz".format(universal_sr))

		# If the file length is shorter than the specified number of seconds that we want, pad the file with zeros to the correct length
		if len(samples) < universal_sr * seconds:

			print("Sound got padded with zeros from {} seconds to {} seconds".format(round(len(samples)/sampling_rate, 3), seconds))

			samples = np.pad(samples, (0, (universal_sr * seconds)-len(samples)))

			# Add the file to the combined sounds
			combined_sounds += samples

		# Else, we take a random point in the file and add the next (5) seconds to the combined sounds
		else:

			# Pick a random number (that is not less than 5 seconds near the end of the sound)
			random_starting_sample = random.randint(0, len(samples) - universal_sr * seconds)

			# Add the sounds in the time domain
			combined_sounds += samples[random_starting_sample:random_starting_sample + universal_sr * seconds]

	# Play the combined sounds
	sd.play(combined_sounds, sampling_rate)
	sd.wait()

	return combined_sounds, labels

""" Takes in a sound sample received from librosa.load and returns the sound samples shifted in amplitude by n_steps."""
def amplitude_shift(samples, n_steps):
	return samples*(n_steps)

""" Visualizes the amplitude shift. Needs the original samples, sampling rate and shifted samples."""
def plot_amplitude(original_samples, shifted_samples, sampling_rate):

	librosa_display.waveplot(original_samples, sr=sampling_rate, label='Original')
	librosa_display.waveplot(shifted_samples, sr=sampling_rate, label='Shifted')

	plt.title('Original vs Shifted samples')
	plt.legend()
	plt.ylabel('Amplitude')
	plt.show()

	return

"""Takes in the samples received from librosa.load and returns samples with noise"""
def add_white_noise(samples, target_snr=2):

	#Calculate the root mean square of the samples
	RMS_samples = np.sqrt(np.mean(samples ** 2))

	#Calculate the root mean square of the noise given a target SNR
	RMS_noise = np.sqrt((RMS_samples ** 2) / 10 ** (target_snr / 10))

	#Generate Additive White Gaussian Noise
	noise = np.random.normal(0, RMS_noise, samples.shape[0])

	#Add noise to samples
	samples += noise

	return samples

""" Takes in a sound sample and sampling rate received from librosa.load and returns the sound samples shifted in frequency (pitch) by n_steps."""
def frequency_shift(samples, sampling_rate, n_steps):
	shifted_samples = librosa.effects.pitch_shift(samples,sampling_rate,n_steps=n_steps)
	return shifted_samples

""" Visualizes the amplitude shift. """
def plot_frequency(samples, shifted_samples, sampling_rate):
	plt.magnitude_spectrum(samples, Fs=sampling_rate, label='Original')
	plt.magnitude_spectrum(shifted_samples, Fs=sampling_rate, label='Shifted')
	plt.legend()
	plt.show()
	return

""" Time-stretches a sample by a given rate, returns time-stretched sample. """
def time_stretch(samples, rate):
	shifted_samples = librosa.effects.time_stretch(samples, rate=rate)
	return shifted_samples

"""Takes in samples and adds random background noise from one of the other files in full_path"""
def add_random_background_noise(samples, sampling_rate):

	original_noise = np.array([0])
	
	#Repeat loading files until noise segment is at least 5 seconds
	while (original_noise.shape[0] < 110250):
		
		#Get the path to a random soundfile
		random_sample_path = df_train['full_path'][np.random.randint(0, len(df_train['full_path']))]
		
		#Load the random sample
		random_sample, sr = librosa.load(random_sample_path)
		
		#Get background noise from random sample
		original_noise = ne.get_noise(random_sample, sr)
		
	#Cut noise to correct format if there is more noise than samples
	if original_noise.shape[0] > samples.shape[0]:
		start_index = np.random.randint(0, original_noise.shape[0] - samples.shape[0])
		noise = original_noise[start_index : start_index + samples.shape[0]]
	
	#Create more noise by repeating it in case samples is longer than noise
	else:
		noise = np.array(list(original_noise) * (samples.shape[0] // original_noise.shape[0]) + list(original_noise)[:samples.shape[0] - ((samples.shape[0] // original_noise.shape[0]) * original_noise.shape[0])])
	
	#Calculate SNR of original samples; aim to get the new sample at roughly the same SNR
	target_snr = np.abs( np.log10( np.abs( ( np.mean(samples) ) / ( np.std(samples) ) ) ) )
	
	#Calculate the required RMS to reach the target SNR when noise and samples are combined
	RMS_required = np.sqrt((np.sqrt(np.mean(samples ** 2)) ** 2) / 10 ** (target_snr / 10))
	
	#Calculate the constant to multiply with noise to reach target SNR
	const = RMS_required / np.sqrt(np.mean(noise ** 2))
	
	#Filter the foreground samples to be used (not sure if necessary)
	samples = ne.filter_sound(samples, sr, verbose=False)
	
	#Combine noise and filtered samples
	samples += const * noise 
	
	sd.play(samples)
	return samples
	
if __name__ == "__main__":
	pass
