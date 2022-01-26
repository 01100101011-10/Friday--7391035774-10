import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

sampling_freq, signal = wavfile.read('random_sound.wav')

print('\nSignal shape:', signal.shape)
print('Datatype:', signal.dtype)
print('Signal duration:', round(signal.shape[0] / float(sampling_freq), 2), 'seconds')

signal = signal / np.power(2, 15)

signal = signal[:50]

time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq)

plt.plot(time_axis, signal, color='black')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')
plt.show()
