import numpy as np
import pandas as pd
import librosa
import random
import warnings
import sounddevice as sd
import data_reading
import birdcodes
import scipy

# A function that prevents warnings when loading in files with librosa
warnings.simplefilter("ignore")

# Add the path of each file to the train.csv
base_dir = data_reading.read_config()
df_train = pd.read_csv(base_dir + "train.csv")

####### !!!!!!!!!!!!!!! ##########
####### Run these two lines below once if you've never run this file before. It adds a filepath to each file in train.csv #########
# df_train['full_path'] = base_dir + "train_audio/" + df_train['ebird_code'] + '/' + df_train['filename']
# df_train.to_csv(base_dir + "train.csv")

class Add_birdsounds():
	
	def __init__(self, train_csv, metrics=["country"], universal_sr=22050, nr_of_files=2, seconds=5):
		
		self.df_train = train_csv
		
		# Check if the df_train has the full_path column
		assert "full_path" in self.df_train.columns
		
		# A list of all the possible metrics to sort by.
		self.class_metrics = ['country', 'rating', 'playback_used', 'channels',
						 'pitch', 'speed', 'species', 'number_of_notes', 
						 'bird_seen', 'sampling_rate', 'type',
						 'volume', 'country', 'length']
		
		self.continuous_metrics = ['date', 'time', 'duration', 'latitude', 'longitude', 'elevation']
		
		# Initiate arguments
		self.metrics = metrics
		self.universal_sr = universal_sr
		self.nr_of_files = nr_of_files
		self.seconds = seconds
	
	
	""" Takes in the full df_train and outputs a dataframe with only 1 randomly choosen entry from a specified metric
	EXAMPLE: let's say metrics=['country'] --> it chooses only files from the randomly choosen country of Potugal."""
	def filter_metadata_by_metrics(self):
		
		# Copy the original dataframe so we can work with it safely
		dataframe = self.df_train
		
		# Pick files from the metadata randomly by the specified metrics
		for metric in self.metrics:
			
			# Create a while loop that only ends if we find a collection of metadata longer than the nr_of_files
			go_on = False
			loops = 0
			while go_on == False:
				
				loops += 1
				
				# Pick a random value from the metric (If it is 'country', pick from: Portugal, Canada, Sweden, etc...)
				random_pick = random.choice( list( df_train[metric] ) )
				
				# If the metadata has enough files to choose from
				if dataframe[dataframe[metric] == random_pick].shape[0] > self.nr_of_files:
					
					print("Randomly picked {}".format(random_pick) )
					
					# Redefine the dataframe to contain only the instances from the randomly picked metric
					dataframe = dataframe[dataframe[metric] == random_pick]
					
					go_on = True
				
				# If the loop goes on for too long, just skip the current metric
				if loops >= 1000:
					
					print("No combinations possible between {} and {}".format(self.metrics[0], self.metrics[1:]))
					
					go_on = True
				
		return dataframe
	
	
	""" Takes in (any part of the) metadata and returns a random choice of n rows. """
	def pick_files_at_random(self, dataframe):
	
		# Check if there ae enough files available to choose from
		if dataframe.shape[0] < self.nr_of_files:
			raise ValueError("Can't choose {} files from {} samples.".format(self.nr_of_files, dataframe.shape[0]))
		
		# Pick random files from selection
		else:
			
			random_metadata = dataframe.sample(n=self.nr_of_files)
			print("Files picked:\n" + str(random_metadata[["ebird_code", "filename"]]))
			return random_metadata
	
	
	def amplitude_shift(self, sound):
		
		return sound
	
	
	def frequency_shift(self, sound):
		
		return sound
	
	
	""" Takes in one or more rows of the metadata and returns the corresponding soundfiles as numpy arrays. """
	def get_sounds(self):
		
		# Get dataframe that is fitlered by a specific metric
		dataframe = self.filter_metadata_by_metrics()
		
		# Pick a number of specified files from the filtered dataframe
		files = self.pick_files_at_random(dataframe)
		
		# Get the labels that accompany the combined sounds
		labels = [birdcodes.bird_code.get(file) for file in files["ebird_code"]]
		
		# Initiate an array of zeros
		combined_sounds = np.zeros(self.universal_sr * self.seconds,)
		
		for file in files["full_path"]:
			
			# Load the samples
			samples, sampling_rate = librosa.load(file)
			
			# Resample the samples to a universal sampling rate
			if sampling_rate != self.universal_sr:
				
				# Resample with Scipy
				samples, sampling_rate = scipy.signal.resample(x=samples, num=int( self.universal_sr * (len(samples)/sampling_rate) ) ), self.universal_sr
				
				print("Sound has been resampled to {} Hz".format(self.universal_sr))
				
			# If the file length is shorter than the specified number of seconds that we want, pad the file with zeros to the correct length
			if len(samples) < self.universal_sr * self.seconds:
				
				print("Sound got padded with zeros from {} seconds to {} seconds".format(round(len(samples)/sampling_rate, 3), self.seconds))
				
				samples = np.pad(samples, (0, (self.universal_sr * self.seconds)-len(samples)))
				
				# Add the file to the combined sounds
				combined_sounds += samples
				
				continue
			
			# Else, we take a random point in the file and add the next (5) seconds to the combined sounds
			else:
				
				# Pick a random number (that is not less than 5 seconds near the end of the sound)
				random_starting_sample = random.randint(0, len(samples) - self.universal_sr * self.seconds)
				
				# Add the sounds in the time domain
				combined_sounds += samples[random_starting_sample:random_starting_sample + self.universal_sr * self.seconds]
		
		# Play the combined sounds
		sd.play(combined_sounds, sampling_rate)
		sd.wait()
		
		return combined_sounds, labels
	
	








if __name__ == "__main__":
	
	add_birdsounds = Add_birdsounds(df_train, metrics=[], universal_sr=22050, nr_of_files=4, seconds=5)
	
	
	for n in range(3):
		add_birdsounds.get_sounds()