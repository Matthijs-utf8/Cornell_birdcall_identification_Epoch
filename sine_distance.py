import numpy as np
import time
import data_reading
import scipy
import math
import sounddevice

def apply_sine_distance(frame):
    # Create frame for Fast Fourier Transform
    frame_size = frame.shape[0]
    fft_hamming_window = scipy.signal.hamming(frame_size)

    # Compute Fast Fourier Transform
    signal = np.fft.fft(
        frame * fft_hamming_window
    )

    # Calculate sine distance for different frequencies    
    radius = 48
    step_size = radius * 2
    sine_distance_window = scipy.signal.hamming(radius * 2 + 1)    

    filtering = np.array([
        sine_distance(signal, sine_distance_window, center, radius)
        for center in range(radius, signal.shape[0] - radius)
    ])

    filtering = np.pad(filtering, (radius,radius), 'constant', constant_values=(0, 0))

    filtered_signal = signal * filtering

    return np.fft.ifft(filtered_signal).astype('float')

def sine_distance(signal, window, center, radius):
    total_squared_distance = 0
    
    window_center = radius

    # Add each squared distance to the total
    for i in range(-radius, radius + 1):
        normalised_signal = (signal[center + i] / signal[center]).real
        normalised_window = window[i + window_center] / window[window_center]
        total_squared_distance += (normalised_signal - normalised_window) ** 2
    
    # Return the square-root of the average
    return math.sqrt(total_squared_distance / (2 * radius + 1))

if __name__ == "__main__":
    frames, sr = data_reading.default_test_frames()

    start_sine_distance = time.time()

    filtered = np.array([
        apply_sine_distance(frames[i])
        for i in range(50)
    ]).flatten()

    print(time.time() - start_sine_distance)

    for _ in range(10):
        print('filtered')
        # Play edited
        sounddevice.play(filtered.flatten(), sr)
        sounddevice.wait()
        
		# Play normal
        print('normal')
        sounddevice.play(np.array(frames[:50]).flatten(), sr)
        sounddevice.wait()

 