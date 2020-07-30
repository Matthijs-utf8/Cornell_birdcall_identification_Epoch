# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:09:24 2020

@author: Matthijs Schrage
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import sklearn
import warnings
import sounddevice as sd
<<<<<<< HEAD
import data_reading
=======
import noisereduce as nr
>>>>>>> 9fd61b21ca2e49809e8a6920a3b591862a7172b7

# A function that prevents warnings when loading in files with librosa
warnings.simplefilter("ignore")

# A helper function for normalizing signals. It helps with viualisation to get everything on the same scale.
def normalize(x, axis=0):
	return sklearn.preprocessing.minmax_scale(x, axis=axis)

# A function for calculating autocorrelation of a signal
def autocorr(x, t=1):
	return np.corrcoef(np.array([x[:-t], x[t:]]))

# Add the path of each file to the train.csv
base_dir = data_reading.read_config() #os.path.join(os.path.expanduser("~"), "Downloads/birdsong-recognition/")
df_train = pd.read_csv(base_dir + "train.csv")

####### !!!!!!!!!!!!!!! ##########
####### Run these two lines below once if you've never run this file before. It adds a filepath to each file in train.csv #########
# df_train['full_path'] = base_dir + "train_audio/" + df_train['ebird_code'] + '/' + df_train['filename']
# pd.to_csv(base_dir + "train.csv")

""" Split a soundwave up in frames """
def get_frames(samples, window_width, stepsize):
	
	nr_of_frames = (len(samples) - window_width + stepsize) // stepsize
	
	frames = np.array([samples[stepsize*n:stepsize*n+window_width] for n in range(nr_of_frames)])
	
	return nr_of_frames, frames

""" Multiply a series of frames with a window function (hammig window)"""
def window_function_transform(frames):
	
	from scipy.signal.windows import hamming
	
	# Construct a Hamming window with the same number of datapoints as 1 frame. 
	window = hamming(frames.shape[1])
	
	# Multiply all frames with our window function
	new_frames = np.array([frame*window for frame in frames])
	
	return new_frames

""" Compute the statistical features of each frame that we need for our noise/non-noise heuristic """
def get_statistcal_features(frames, heuristic="energy"):
	
	# Initiate the necessary lists where we want to store our features
	energies = np.array([np.sum(frame**2) for frame in frames])
	
	return energies, np.mean(energies)

""" Helps us with the automatic calculation of the energy coeff. for each file. The higher the S/N is, the higher the energy coeff. is. """
def compute_energy_coefficient(samples, base_coefficient=1):
	
	# Compute an approximation of the Signal-to-Noise-Ratio
	SNR = np.abs( np.log10( np.abs( ( np.mean(samples) ) / ( np.std(samples) ) ) ) )
	
	# Compute the energy coefficient
	base_coefficient = base_coefficient
	energy_coefficient = base_coefficient * ( ( 1 / (SNR) ) ** 2 )
	
	return SNR, energy_coefficient

""" The function that bring it all together. """
def get_noise_frames(samples, sampling_rate, window_width=2048, stepsize=512, verbose=False):
	
	""" Preparation for separating pure noise from non-pure noise. """
	
	# Separate the samples in frames according to the window_width and stepsize
	nr_of_frames, frames = get_frames(samples, window_width=window_width, stepsize=stepsize)
	
	# Use a window function (hamming works best) on all our frames
	frames = window_function_transform(frames)
	
	# Get the statistical features that we need. For now only 'energy' works.
	energies, mean_energy = get_statistcal_features( frames )
	
	# Get the energy coefficient that we need for separating pure noise from non-pure noise.
	SNR, energy_coefficient = compute_energy_coefficient(samples, base_coefficient=2)
	
	print("Energy coefficient: " + str(round(energy_coefficient, 3) ) )
	print("Signal-to-Noise: " + str(round(SNR, 3)))
	
	""" Separating pure noise from non-pure noise. """
	
	# Initiate lists to store the separated frames in.
	noisy_frames = []
	non_noisy_frames = []
	noisy_energy = []
	non_noisy_energy = []
	
	# Go through all of the frame-energies. The ones below a certain threshold have a very high chance of being pure background noise.
	for index, energy in enumerate(energies):
		
		if energy < energy_coefficient * mean_energy:
			
			# Add the pure noisy parts to the appropriate list
			noisy_frames.extend(frames[index][int((window_width-stepsize)/2):int((window_width+stepsize)/2)])
			noisy_energy.append(energy)
		
		else:
			# Add the non-noise frames to the appropriate list
			non_noisy_frames.extend(frames[index][int((window_width-stepsize)/2):int((window_width+stepsize)/2)])
			non_noisy_energy.append(energy)
	
	# A measure for how well the noise is predictable (higher is better). The better predictable it is, the better a spectral noise gate will work
	print("Noise predictability: " + str(round(autocorr(noisy_frames)[0,1] / autocorr(non_noisy_frames)[0,1], 3) ) )
	
	""" Plotting """
	
	if verbose == True:
		
		# Initiate time domain axes for some different graphs
		t_soundwave = np.linspace(0, len(samples)/sampling_rate, len(samples))
		t_soundwave_noisy = np.linspace(0, len(noisy_frames)/sampling_rate, len(noisy_frames))
		t_soundwave_non_noisy = np.linspace(0, len(non_noisy_frames)/sampling_rate, len(non_noisy_frames))
		
		t_windowed_features = np.linspace(0, len(samples)/sampling_rate, nr_of_frames)
		t_windowed_features_noisy = np.linspace(0, len(noisy_frames)/sampling_rate, len(noisy_energy))
		t_windowed_features_non_noisy = np.linspace(0, len(non_noisy_frames)/sampling_rate, len(non_noisy_energy))
		
		# Plot the signal versus the signal energy
		plt.figure(figsize=(20,12))
		plt.title("Energy whole signal")
		plt.plot(t_soundwave, normalize(samples), alpha=0.5)
		plt.plot(t_windowed_features, normalize(energies))
		plt.show()
		
		# Plot the signal versus the signal energy
		plt.figure(figsize=(20,12))
		plt.title("Energy pure noise signal")
		plt.plot(t_soundwave_noisy, normalize(noisy_frames), alpha=0.5)
		plt.plot(t_windowed_features_noisy, normalize(noisy_energy) )
		plt.show()
		
		# Plot the signal versus the signal energy
		plt.figure(figsize=(20,12))
		plt.title("Energy non pure noise signal")
		plt.plot(t_soundwave_non_noisy, normalize(non_noisy_frames), alpha=0.5)
		plt.plot(t_windowed_features_non_noisy, normalize(non_noisy_energy))
		plt.show()
	
	return np.array(noisy_frames)

def filter_sound(fullpath, verbose=False):
	
	# Read in the audiofile with librosa.
	samples, sampling_rate = librosa.load(fullpath)
	
	noise = get_noise_frames(samples=samples, sampling_rate=sampling_rate, verbose=verbose)
	
	reduced_noise = nr.reduce_noise(audio_clip=samples, noise_clip=noise, verbose=verbose)
	
	if verbose == True:
		
		print("Playing original samples")
		sd.play(samples, sampling_rate)
		sd.wait()
		
		print("PLaying noise")
		sd.play(noise, sampling_rate)
		sd.wait()
		
		print("Playing reduced noise samples")
		sd.play(reduced_noise, sampling_rate)
		sd.wait()

if __name__ == "__main__":
	filter_sound(df_train['full_path'][2], verbose=True)
