import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import Noise_Extractor
import sklearn
import pydub


""" A helper function for normalizing signals. Nice for plotting and for normalizing the dataset. """
def normalize(samples, axis=0):
	if len(samples.shape) == 1:
		start_shape = samples.shape
		return sklearn.preprocessing.normalize(samples.reshape( (-1,1) ), axis=axis, norm="max").reshape(start_shape)
	else:
		return sklearn.preprocessing.normalize(samples, axis=axis, norm="max")


""" A function for resampling any signal. Automatically checks if the sampling rat is already correct. """
def resample(samples, sampling_rate, universal_sr):

	if sampling_rate == universal_sr: return samples

	else: return scipy.signal.resample(x=samples, num=int( universal_sr * (len(samples)/sampling_rate) ) )


""" A function for converting an array to a .mp3 file """
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


""" A function call to Noise_Extractor.py """
def extract_noise(samples, sampling_rate, window_width=2048, stepsize=512, verbose=False):

	return Noise_Extractor.filter_sound(samples, sampling_rate, window_width=window_width, stepsize=stepsize, verbose=verbose)


""" A function to make a specific spectrogram """
def make_spectrogram(samples, window_width, spectrogram="normal", verbose=False):

	if spectrogram == "normal":

		spectr = np.abs(librosa.stft(samples, win_length=512, window="hamm"))

		dB_spectr = librosa.amplitude_to_db(spectr, ref=np.max)

		if verbose:

			plt.imshow(spectr)
			plt.show()

			plt.imshow(dB_spectr)
			plt.show()

		return dB_spectr

	elif spectrogram == "mel":

		mel_spectr = librosa.feature.melspectrogram(samples, win_length=512, window="hamm")

		mel_dB_spectr = librosa.amplitude_to_db(mel_spectr, ref=np.max)

		if verbose:

			plt.imshow(mel_spectr)
			plt.show()

			plt.imshow(mel_dB_spectr)
			plt.show()

		return mel_dB_spectr

	else:

		raise  ValueError("{} does not exist".format(spectrogram))
