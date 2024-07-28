import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import cheby2, lfilter
from scipy.io import wavfile
import simpleaudio as sa
from frequency_domain_adaptive_filters.fdaf2 import fdaf2

# Parameters
M = 4001
fs = 8000  # Sampling rate (voice frequency ranges from 300 - 3400 Hz)

# Create RIR using Chebyshev's filter like in MatLAB
b, a = cheby2(4, 20, [0.1, 0.7], btype='bandpass', fs=fs)

# Generate random signal
random_signal = np.log(0.99 * np.random.rand(M) + 0.01) * np.sign(np.random.randn(M)) * np.exp(-0.002 * np.arange(M))

# Filter the random signal
H = lfilter(b, a, random_signal)

# Normalize the Room Impulse Response
H = H / np.linalg.norm(H) * 4

# Load the echoed signal
input_signal_path = 'samples/Hello_Echoe.wav'
sample_rate, mySig = wavfile.read(input_signal_path)

# Normalize the input signal
mySig = mySig / np.max(np.abs(mySig))

# Filter with room impulse response
dhat = lfilter(H, 1, mySig)

# Plotting the echoed signal
t = np.arange(len(mySig)) / sample_rate
plt.figure(figsize=(10, 4))
plt.plot(t, mySig, label='Original Signal')
plt.plot(t, dhat, label='Echoed Signal')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Hello Echo Signal')
plt.legend()
plt.grid(True)
plt.show()

# Play the echoed signal
echoed_signal = (dhat * 32767).astype(np.int16)
play_obj = sa.play_buffer(echoed_signal, 1, 2, sample_rate)
play_obj.wait_done()

# LMS parameters
mu = 0.025  # Step size
filter_length = 2048
W0 = np.zeros(filter_length)
del_param = 0.01
lam = 0.98

# Ensure the signal lengths are divisible by the filter length
x = mySig[:len(mySig) - len(mySig) % filter_length]
d = dhat[:len(dhat) - len(dhat) % filter_length]

y, e = fdaf2(x, d, mu, filter_length, del_param, lam)

# Plotting
t = np.arange(len(e)) / sample_rate
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, d[:len(e)], 'b')
plt.axis([0, 5, -1, 1])
plt.ylabel('Amplitude')
plt.title('Hello Echo Signal')
plt.subplot(2, 1, 2)
plt.plot(t, e, 'r')
plt.axis([0, 5, -1, 1])
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Output of Acoustic Echo Canceller (mu = 0.025)')
plt.tight_layout()
plt.grid(True)
plt.show()

# Play the filtered signal
filtered_signal = (e * 32767).astype(np.int16)
play_obj = sa.play_buffer(filtered_signal, 1, 2, sample_rate)
play_obj.wait_done()