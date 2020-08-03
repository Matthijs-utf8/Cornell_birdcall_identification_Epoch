import data_reading
import sounddevice

metadata = data_reading.get_train_metadata()

frames1, sample_rate = data_reading.get_frames_from_index(3, metadata)
frames2, _ = data_reading.get_frames_from_index(250, metadata)

sounddevice.play(frames1[:50].flatten() + frames2[:50].flatten(), sample_rate)
sounddevice.wait()