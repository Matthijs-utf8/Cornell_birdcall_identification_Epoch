import numpy as np
import pandas as pd
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
def create_shuffled_dataset(metrics, files_to_combine, universal_sr, clip_seconds):
		
		
	new_dataframe = sound_shuffling.filter_metadata_by_metrics(df_train, metrics=metrics, nr_of_files=files_to_combine)
	
	
	random_files = sound_shuffling.pick_files_at_random(new_dataframe, nr_of_files=files_to_combine)
	
	
	combined_file, labels = sound_shuffling.combine_files(files=random_files, universal_sr=universal_sr, seconds=clip_seconds)
		
	return combined_file, labels 

"""Creat actual dataset with spectrograms"""
if __name__ == "__main__":
	
	# Create a dictionary for storing the labels that accompany the files
	### !!!!!!!!!!!!!!!! ### 
	### LABELING WERKT NOG NIET ###
	all_labels = {}
	
	""" Hyperparameters """
	dataset_size = 1
	metrics = ["country"] # If list is empty, it chooses from all the files
	files_to_combine = 2 # Number of files to merge each time
	universal_sr = 22050
	clip_seconds = 5 # The length of the new clips in seconds
	window_width = 512 #
	
	for n in range(dataset_size):
	
		combined_audio, labels = create_shuffled_dataset(
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
		
		spectr = preprocessing.make_spectrogram(
												combined_audio, 
												window_width=512, 
												spectrogram="normal", 
												verbose=False
												)
		
		PIL_img = Image.fromarray(spectr)
		
		PIL_img.save(base_dir + "/two_combined_birdsounds_by_country/" + str(n) + ".jpg")
		
		### HIER NOG LABEL TOEVOEGEN AAN DICTIONARY ###
	
	
	