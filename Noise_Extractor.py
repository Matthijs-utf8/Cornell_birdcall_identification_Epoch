# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:09:24 2020

@author: Matthijs Schrage
"""

"""
Samenvatting van de file:
	
	Een file die een audio bestand laadt en de pure noise kan scheiden van de non-pure noise.
	Als we eenmaal van elk bestand pure-noise hebben 
		kunnen we heel simpel een spectral noise gate gebruiken om de noise te filteren.
		
	De heuristic waar ik nu mee aan het spelen ben is energy, omdat dat de meest gebruikte en belangrijkste is.
	We moeten echter een veel beter heuristic gebruiken om dit echt goed te laten werken.
	
	Works in progress:
		- Energy coefficient automatisch berekenen
		- Heuristic verbeteren
		- Een validation methode bedenken waarmee we verschillende heuristics kunnen vergelijken op prestatie
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from librosa.feature import zero_crossing_rate, spectral_centroid, spectral_rolloff, spectral_bandwidth, rms
from librosa.onset import onset_detect, onset_backtrack, onset_strength
import sklearn
import warnings
from scipy.signal.windows import hamming
import sounddevice as sd

# A function that prevents warnings when loading in files with librosa
warnings.simplefilter("ignore")

# A helper function for normalizing signals. It helps with viualisation to get everything on the same scale.
def normalize(x, axis=0):
	return sklearn.preprocessing.minmax_scale(x, axis=axis)

# A function for calculating autocorrelation of a signal
def autocorr(x, t=1):
	return np.corrcoef(np.array([x[:-t], x[t:]]))

# Add the path of each file to the train.csv
base_dir = os.path.join(os.path.expanduser("~"), "Downloads/birdsong-recognition/")
df_train = pd.read_csv(base_dir + "train.csv")

####### !!!!!!!!!!!!!!! ##########
####### Run these two lines below once if you've never run this file before. It adds a filepath to each file in train.csv #########
# df_train['full_path'] = base_dir + "train_audio/" + df_train['ebird_code'] + '/' + df_train['filename']
# pd.to_csv(base_dir + "train.csv")

###### Hyper parameters ######

plotting = True

### Data ###
y, sr = librosa.load(df_train['full_path'][3])

#Gives a measure of how much noise is in the file
autocorrelation = autocorr(y)[0,1]
print("Autocorrelation: " + str(autocorrelation))
# sd.play(y, sr)

### Window function ###
window_width = 2048
stepsize = 2048
if window_width != stepsize:
	raise ValueError("De functie werkt alleen (nog) als deze twee hetzelfde zijn")
# Hamming window has proven to be the best window for birdsounds from the papers I (Matthijs) have read
window = hamming(window_width)

##############################

# Calculate the frames. Skips the data at the end of the last frame does not fit perfectly.
nr_of_frames = (len(y) - window_width + stepsize) // stepsize
frames = np.array([y[stepsize*n:stepsize*n+window_width] for n in range(nr_of_frames)])

# Initiate the necessary lists where we want to store our features
energies = []
zero_crossing_rates = []

# For every frame, compute the features of the frame
for frame in frames:
	
	# Caculate the newly windowed frame
	new_frame = frame * window
	
	# Calculate the energy for the newly windowed frame
	window_energy = np.sum(new_frame**2)
	
	# Calculate the zero crossing rate for the newly windowed frame
	zcr = np.sum(librosa.core.zero_crossings(new_frame))/window_width
	
	# Append the features that we want
	energies.append(window_energy)
	zero_crossing_rates.append(zcr)

# Get the mean energy of the signal
mean_energy = np.mean(energies)

# Initiate lists for storing non_birdcall and birdcall sounds
noisy_frames = []
birdcall_frames = []

################
# We will have to think of a better heuristic than just the energy of a signal
# Possible other candidates: zero_crossing_rate, spectral
################
for index, energy in enumerate(energies):
	
	coeff = 0.05
	
	############ 
	# AUTOMATISEREN van de coefficient. 
	# Ik zit te denken aan de coefficient afleiden van autocorrelatie. 
	# Die twee lijken elkaar te beïnvloeden.
	# De optimale coefficienten tussen files kunnen verschillen van 0.02 tot 0.3 ofzo
	# Ik vermoed dat het te maken heeft met de Signal-to-Noise-Ratio (SNR)
	# Ik bepaal hem nu elke keer met manual, door te luisteren naar de "noisy_frames"
	################
	if energy < coeff * mean_energy:
		
		#Add the pure noisy frames to the appropriate list
		noisy_frames.extend(frames[index])
	
	else:
		#Add the non-noise frames to the appropriate list
		birdcall_frames.extend(frames[index])

#############
# Dit stuk met de fourier transforms gebruik ik om de noisy en non noisy frames
# met elkaar te vergelijken. Weet niet of dit zo gaat lukken,
# maar ik probeer door verbanden te vinden een optimale manier te bereiken
# om pure noise van non pure noise te scheiden.
#############
print(autocorr(noisy_frames)[0,1])
print(autocorr(birdcall_frames)[0,1])

fft_samples_noise = np.fft.fft(noisy_frames, n=200000)
fft_samples_birds = np.fft.fft(birdcall_frames, n=200000)

print(np.corrcoef(fft_samples_noise, fft_samples_birds).real[0,1])

frequencies_noise = np.abs(np.fft.fftfreq(len(fft_samples_noise), d=1/sr))
frequencies_birds = np.abs(np.fft.fftfreq(len(fft_samples_birds), d=1/sr))

if plotting == True:
	### Plotting stuff ###
	t_soundwave = np.linspace(0, len(y)/sr, len(y))
	t_windowed_features = np.linspace(0, len(y)/sr, nr_of_frames)
	
	#Plot the signal versus the signal energy
	plt.figure(figsize=(20,12))
	plt.title("Energy")
	plt.plot(t_soundwave, y, alpha=0.5)
	plt.plot(t_windowed_features, normalize(energies))
	plt.show()
	
	#Plot the signal versus the zero crossing rate
	plt.figure(figsize=(20,12))
	plt.title("Zero crossing rate")
	plt.plot(t_soundwave, y, alpha=0.5)
	plt.plot(t_windowed_features, normalize(zero_crossing_rates))
	plt.show()
	
	plt.figure(figsize=(20,12))
	plt.title("Noise fft")
	plt.plot(frequencies_noise, fft_samples_noise)
	plt.show()
	
	plt.figure(figsize=(20,12))
	plt.title("Birdsounds fft")
	plt.plot(frequencies_birds, fft_samples_birds)
	plt.show()

# Uncomment deze regel als je de noisy frames wil horen afspelen
# sd.play(noisy_frames, sr)

