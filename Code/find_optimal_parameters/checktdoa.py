from tdoa import tdoa, tdoa_prepare_x
# remove, just for convenience 
# Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse

from refsignal import refsignal            # model for the KITT audio beacon signal
from wavaudioread import wavaudioread
from recording_tool import recording_tool
recordings_path = "./Finished recordings/"

Fs_RX = 48000
Lhat = 2000
start_threshold = 0.1552020202020202
file = "recording-beacon-50cm3.wav"
Fs, x = wavfile.read('audio-beacon.wav')
y = wavaudioread(recordings_path+file, Fs_RX)
print(len(y))
print(len(x))
print(Fs_RX)

x = tdoa_prepare_x(x, start_threshold)
print(len(y))
result = tdoa(x, y, Lhat, method="ch2", start_threshold=start_threshold, Fs_RX=Fs_RX, file=file)


print(result)

print(len(y))
print(len(x))
print(Fs_RX)