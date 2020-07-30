"""
@author: Jesse Maas
"""

import numpy as np
import time
import data_reading
import scipy
import math
import sounddevice

# Applies a filter to the sound based on the sine_distance heuristic
def apply_sine_distance(frame):
    # Create window for Fast Fourier Transform
    frame_size = frame.shape[0]
    fft_hamming_window = scipy.signal.hamming(frame_size)

    # Compute Fast Fourier Transform
    signal = np.fft.fft(
        frame# * fft_hamming_window
    )

    radius = 8 # Decides how many frequencies around the center are considered in the sine_distance calculation
    sine_distance_window = scipy.signal.hamming(radius * 2 + 1)    

    # Construct a filter based on the sine_distance heuristic
    filtering = np.array([
        sine_distance(signal, sine_distance_window, center, radius)
        for center in range(radius, signal.shape[0] - radius)
    ])

    # All sounds with a sine_distance less than the average are considered noise
    avg = np.average(filtering)
    filtering = (filtering > avg).astype('float') # create a filter of 0.0 and 1.0 values
    filtering = np.pad(filtering, (radius, radius), 'constant', constant_values=(0, 0))

    # Apply filter
    filtered_signal = signal * filtering

    # Return the inverse fourier transform of the filtered sounds
    return np.fft.ifft(filtered_signal).real

def sine_distance(signal, window, center, radius):
    window_center = radius

    # Normalise the signal and window function such that at the center the strength is 1
    normalised_signal = np.absolute(signal[center - radius : center + radius + 1]) / np.absolute(signal[center])
    normalised_window = window / window[window_center]

    distance = normalised_signal - normalised_window
    
    total_squared_distance = np.dot(distance, distance)
    
    # Return the square-root of the average squared distance
    return math.sqrt(total_squared_distance / (2 * radius + 1))

if __name__ == "__main__":
    frames, sr = data_reading.default_test_frames()
    """ 
    print(frames.shape)
    
    avg = np.average(frames[:50])
    frames += np.random.rand(frames.shape[0]) * avg * 0.5

    """
    test_frame_length = 100

    start_sine_distance = time.time()

    filtered = np.array([
        apply_sine_distance(frames[i])
        for i in range(test_frame_length)
    ]).flatten()

    print(time.time() - start_sine_distance)

    for _ in range(10):
        # Play edited
        print('filtered')
        sounddevice.play(filtered.flatten(), sr)
        sounddevice.wait()
        
        # Play normal
        print('normal')
        sounddevice.play(np.array(frames[:test_frame_length]).flatten(), sr)
        sounddevice.wait()
        
 