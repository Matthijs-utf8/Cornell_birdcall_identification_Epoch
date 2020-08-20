import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import Noise_Extractor
import sklearn
import pydub
import data_reading
import pandas as pd

# Add the path of each file to the train.csv
base_dir = data_reading.read_config()
df_train = pd.read_csv(base_dir + "train.csv")


""" A helper function for normalizing signals. Nice for plotting and for normalizing the dataset. """
def normalize(samples, axis=0):

	""""
	:param samples: numpy.array, array of (x-dimensional) samples
	:param axis: int, the axis that is normalized
	:return: numpy.array, array of normalized samples
	"""

	# If the data is one dimensional:
	if len(samples.shape) == 1:

		# We need to use some weird reshaping to get this to work, but it works.
		start_shape = samples.shape
		return sklearn.preprocessing.normalize(samples.reshape((-1, 1)), axis=axis, norm="max").reshape(start_shape)

	# If the data is multidimensional:
	else:
		return sklearn.preprocessing.normalize(samples, axis=axis, norm="max")


""" A function for resampling any signal. Automatically checks if the sampling rat is already correct. """
def resample(samples, sampling_rate, universal_sr=22050):

	""""
	:param samples: numpy.array, 1d array of samples (preferably from librosa.load())
	:param sampling_rate: int, the sampling rate at which the sound was recorded
	:param universal_sr: int, the sampling rate at which the sound should be resampled
	:return: numpy.array, A 1d numpy array of resampled sound at the universal sampling rate
	"""

	# If the frequency is already the same as the universal frequency, just skip it.
	if sampling_rate == universal_sr:
		return samples
	# Else, resample the sound with scipy.
	else:
		return scipy.signal.resample(x=samples, num=int(universal_sr * (len(samples)/sampling_rate)))


""" A function for converting an array to a .mp3 file """
def write(f, x, sr=22050, normalized=True):

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


""" A function call to Noise_Extractor.py """
def extract_noise(samples, sampling_rate, window_width=2048, step_size=512, verbose=False):

	""""
		:param samples: 		numpy.array, 1d array of samples (preferably from librosa.load())
		:param sampling_rate: 	int, the sampling rate at which the sound was recorded
		:param window_width: 	int, power of 2,	the width of the window that is used to extract the features from the samples.
												the features are used to discern between pure noise and non-pure noise.
		:param step_size: 		int, power of 2, the number of samples that the window moves each step.
		:param verbose: 		bool, set to True if you want to see graphs and text about the results.
		:return: 				numpy.array, same soundclip as before, but with removed noise
	"""

	return Noise_Extractor.filter_sound(samples, sampling_rate, window_width=window_width, stepsize=step_size, verbose=verbose)


def cut_spectrogram(spectrogram, spectrogram_slices_per_input):

	""""
		:param spectrogram:						numpy.array, A 2d numpy array depicting a spectrogram
		:param spectrogram_slices_per_input: 	int, 	The number of slices that make up a 5 second spectrogram. Calculate with the following formula:
														int(seconds * (np.ceil(sampling_rate / hop_length)))
		:return: 								numpy.array, 	A 3d array that contains the spectrogram slices. If the sound does not have a duration
																that is dividable by 5, the last few seconds will be lost.
	"""

	# Split up into slices of (by default) 5 seconds
	n_fragments = spectrogram.shape[1] // spectrogram_slices_per_input
	slices = np.zeros((n_fragments, spectrogram.shape[0], spectrogram_slices_per_input))

	for i in range(n_fragments):
		begin, end = i * spectrogram_slices_per_input, (i + 1) * spectrogram_slices_per_input
		slices[i] = spectrogram[..., begin:end]

	return np.array(slices)


""" A function to make a specific spectrogram """
def make_spectrogram(samples, sampling_rate=22050, seconds=5, window_width=512, spectrogram_type="normal", verbose=False):

	"""
		:param samples: 			numpy.array, 1d array of samples (preferably from librosa.load())
		:param sampling_rate: 		int, the sampling rate at which the sound was recorded
		:param seconds: 			int, the number of seconds that the user wants to soundclips to last
		:param window_width: 		int, number to the power of 2, the number of samples that is used for the short-time-fourier-transform
		:param spectrogram_type: 	str, 'normal' or 'mel'
		:param verbose: 			bool, set to True if you want to see graphs and text about the results.
		:return: 					numpy.array, 	A 3d array that contains the spectrogram slices. If the sound does not have a duration that is dividable by 5,
													the last few seconds will be lost.

	"""
	# Define the hop_length with this formula and calculate the number of slices per 5 second clip
	hop_length = int(window_width/2)
	spectrogram_slices_per_input = int(seconds * (np.ceil(sampling_rate / hop_length)))

	if spectrogram_type == "normal":

		# Get the spectrogram
		spectr = np.abs(librosa.stft(samples, n_fft=window_width, hop_length=hop_length, win_length=window_width, window="hamm", center=True))

		# Rescale it to a logarithmic scale
		dB_spectr = librosa.amplitude_to_db(spectr, ref=np.max)

		# Show the spectrograms
		if verbose:

			plt.imshow(spectr)
			plt.show()

			plt.imshow(dB_spectr)
			plt.show()

		# Return the rescaled and sliced spectrogram
		return cut_spectrogram(dB_spectr, spectrogram_slices_per_input)

	elif spectrogram_type == "mel":

		# Get the spectrogram
		mel_spectr = librosa.feature.melspectrogram(samples, n_fft=window_width, hop_length=hop_length, win_length=window_width, window="hamm", center=True)

		# Rescale it to a logarithmic scale
		mel_dB_spectr = librosa.amplitude_to_db(mel_spectr, ref=np.max)

		# Show the spectrograms
		if verbose:

			plt.imshow(mel_spectr)
			plt.show()

			plt.imshow(mel_dB_spectr)
			plt.show()

		# Return the rescaled and sliced spectrogram
		return cut_spectrogram(mel_dB_spectr, spectrogram_slices_per_input)

	else:

		raise ValueError("{} does not exist".format(spectrogram))

if __name__ == "__main__":
	sounds, sample_r = librosa.load(df_train["full_path"][3])
	make_spectrogram(samples=sounds, spectrogram_type="normal", verbose=True)
