# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:24:49 2020

@author: Matthijs Schrage
"""

import os
import numpy as np
import scipy
import warnings
import pydub

warnings.simplefilter("ignore")

### A FUNCTION FOR READING MP3 FILES AND CONVERTING THEM TO AN ARRAY ###
def read(file_path, normalized=True):
	
	# Read the file with pydub
	a = pydub.AudioSegment.from_mp3(file_path)
	
	# Get the samples from the file
	y = np.array(a.get_array_of_samples())
	
	# If the audio has 2 channels, reshape the array
	if a.channels == 2:
		y = y.reshape((-1, 2))
	
	# If we're working with normalized audio, divide all values by 2**15 to scale between -1 and 1
	if normalized:
		return a.frame_rate, np.float32(y) / 2**15
	
	# Normal unnormalized retrun
	else:
		return a.frame_rate, y

### A FUNCTION FOR CONVERTING AN ARRAY TO AN MP3 FILE ###
def write(f, sr, x, normalized=True):
	
	# Check the amount of channels that we need to convert the file to
	channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
	
	# If the array is normalized, scale the array by 2**15, else do nothing to the array.
	if normalized:  # normalized array - each item should be a float in [-1, 1)
		y = np.int16(x * 2 ** 15)
	else:
		y = np.int16(x)
	
	# Convert the array to mp3 with pydub.
	song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
	song.export(f, format="mp3", bitrate="64k")

### A FUNCTION FOR RESAMPLING A FILE TO THE SPECIFIEC SAMPLING RATE ###
def resample(dataset_root, 
				 filename,
				 resampling_rate=32000,
				 write_new=False):
	"""
	dataset_root: location/path of the file
	filename: plain filename, no path included
	resampling rate: desired sampling rate of the resampled output
	write_new: specify if you want to rewrite the file or if you just want the output of the function
	"""
	
	print("Resampling {} ".format(filename))
	
	# Read the file
	sampling_rate, samples = read(dataset_root + "/" + filename, normalized=True)
	
	# If the file does not need to be resampled, immediately return the same samples
	if sampling_rate == resampling_rate:
		return sampling_rate, samples
	
	print("Sampling rate before: {}".format(sampling_rate))
	
	# Resample the samples
	new_samples = scipy.signal.resample(x=samples, num=int( resampling_rate * (len(samples)/sampling_rate) ) )
	
	# If we want to overwrite the old mp3 file with this new one, use write_new = True
	if write_new == True:
		write(dataset_root + "/"  + filename, resampling_rate, new_samples, normalized=True)
		print("Overwritten {}".format(filename))
		return resampling_rate, new_samples
	
	# Else, just retrun the samples
	else:
		return resampling_rate, new_samples


if __name__ == "__main__":
	
	dataset_root = os.path.join(os.path.expanduser("~"), "Downloads/birdsong-recognition/train_audio/labeled_data")
	
	nr_of_files = len(os.listdir(dataset_root))
	nr=1

	for filename in os.listdir(dataset_root):
		
		resample(dataset_root, filename, write_new=False)
		print("{} / {}".format(nr, nr_of_files))
		nr += 1
