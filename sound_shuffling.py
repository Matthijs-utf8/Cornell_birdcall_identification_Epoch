# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:36:59 2020

@author: Matthijs Schrage
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import sklearn
import random
import warnings
import sounddevice as sd
import noisereduce as nr
import re

random.seed(4)

# A function that prevents warnings when loading in files with librosa
warnings.simplefilter("ignore")

# Add the path of each file to the train.csv
base_dir = os.path.join(os.path.expanduser("~"), "Downloads/birdsong-recognition/")
df_train = pd.read_csv(base_dir + "train.csv")

# Make a place in memory where we store all the different species
all_ebird_codes = list( set( df_train.ebird_code.tolist() ) )

""" A function for adding files together. Only one method now, but we should have mroe than one to see what works best."""
def add_files(files, method="add_to_shortest"):
	
	# Get the number of files
	nr_of_files = len(files)
	
	# Add a random sample of the longer files to the shortest one
	if method == "add_to_shortest":
		
		# Get the lentgths of the file, the new file will be the length of the shortest file
		lengths = [len(files[n]) for n in range(nr_of_files)]
		shortest = np.min(lengths)
		
		# Initialise new file
		new_file = np.zeros(shortest,)
		
		# Add the files together. Choose a random point in the longer files to add to the shortest one.
		for file in files:
			random_number = random.randint(0, len(file)-shortest)
			new_file += file[random_number:random_number+shortest]
	
	else:
		raise ValueError("Method '{}' does not exist".format(method))
	
	return new_file

""" Takes in a part of the metadata and returns a random choice of n loaded files. """
def pick_and_read_random_file(dataframe, nr_of_files):
	
	# Pick random files
	random_metadata = dataframe.sample(n=nr_of_files)
	paths = random_metadata["full_path"]
	
	print("Files picked:\n" + str(random_metadata[["ebird_code", "filename"]]))
	
	# Read the files
	random_files = np.array([librosa.load(path)[0] for path in paths])
	
	return random_files

""" A function for selecting files based on certain metrics"""
def select_files(nr_of_files, metric="random"):
	
	#Get a sampling rate to choose from. We can only add files together with the same sampling rate (if we don't want to resample)
	random_sr = random.choice( list( set( df_train['sampling_rate'] ) ) )
	
	# Define a dataframe from df_train with only 1 type of sampling rate
	df = df_train[df_train['sampling_rate'] == random_sr]
	
	if metric == "random":
		
		random_files = pick_and_read_random_file(dataframe=df, nr_of_files=nr_of_files)
		
		new_file = add_files(random_files)
		
		#Play the new file
		sd.play(new_file, int(re.findall("[0-9]{2,}", random_sr)[0]))
	
	elif metric == "country":
		
		# Define a dataframe from df with only 1 country
		random_country = random.choice( list( set( df['country'] ) ) )
		df = df[df['country'] == random_country]
		
		random_files = pick_and_read_random_file(dataframe=df, nr_of_files=nr_of_files)
		
		new_file = add_files(random_files)
		
		#Play the new file
		sd.play(new_file, int(re.findall("[0-9]{2,}", random_sr)[0]))
	
	elif metric == "location":
		
		# Define a dataframe from df with only 1 location
		random_location = random.choice( list( set( df_train['location'] ) ) )
		df = df[df['location'] == random_location]
		
		random_files = pick_and_read_random_file(dataframe=df, nr_of_files=nr_of_files)
		
		new_file = add_files(random_files)
		
		#Play the new file
		sd.play(new_file, int(re.findall("[0-9]{2,}", random_sr)[0] ) )
		
	else:
		
		raise ValueError("Metric '{}' does not exist".format(metric))
			
select_files(4, metric="country")