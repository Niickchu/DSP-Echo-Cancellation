# Copyright 2020 ewan xu<ewan_xu@outlook.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import librosa
import soundfile as sf
import pyroomacoustics as pra

from frequency_domain_adaptive_filters.fdaf import fdaf

import matplotlib.pyplot as plt
import librosa.display


def main():

  print("Creating Echo Signal...")
  x, sr  = librosa.load('samples/female.wav',sr=8000)
  d, sr  = librosa.load('samples/male.wav',sr=8000)

  rt60_tgt = 1.5   # desired reverberation time (the time in seconds for the energy to drop by 60 dB)
  room_dim = [5, 5, 5]

  e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
  room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
  room.add_source([1.5, 1.5, 1.5])
  room.add_microphone([0.1, 0.5, 0.1])
  room.compute_rir()
  rir = room.rir[0][0]  #rir = room impulse response
  rir = rir[np.argmax(rir):]  

  y = np.convolve(x,rir)  #x[n] * h[n] = echo component of final signal
  scale = np.sqrt(np.mean(x**2)) /  np.sqrt(np.mean(y**2))
  y = y*scale

  L = max(len(y),len(d))
  y = np.pad(y,[0,L-len(y)])
  d = np.pad(d,[L-len(d),0])
  x = np.pad(x,[0,L-len(x)])

  # d = d + y   #echo signal added to male voice

  # print(x.shape, d.shape)

  #x = female voice non echoed
  #d = female voice echoed with male voice non echoed

  sf.write('samples/original_signal.wav', x, sr, subtype='PCM_16')
  sf.write('samples/echoed_signal.wav', y, sr, subtype='PCM_16')

  print("processing FDAF...")

  e = fdaf(x, y, M=128, mu=0.5)
  sf.write('samples/fdaf_filtered_signal.wav', e, sr, subtype='PCM_16')

  print("Done")

  #fdaf causes e to be longer than d and x, so we pad d and x with zeroes at the end
  x_matched, e_matched = match_lengths(x, e, pad=False)
  y_matched, _ = match_lengths(y, e, pad=False)

  # The performance of an echo canceller is measured in Echo Return Loss Enhancement (ERLE)
  # https://en.wikipedia.org/wiki/Echo_suppression_and_cancellation

  erle = 10 * np.log10(np.mean(y_matched ** 2) / np.mean((y_matched - e_matched) ** 2))

  print("ERLE = " + np.array2string(erle))

  #also save time domain signals of before + after echo cancel
  #also save frequency domain -> Spectrogram

  #time domain
  plot_signals(x=x_matched, y=y_matched, e=e_matched, sr=sr)


def match_lengths(s1, s2, pad=True):
    l1 = len(s1)
    l2 = len(s2)


    if l1 > l2:
        if pad:
          s2 = np.pad(s2, (0, l1 - l2))
        else:
          s1 = s1[:l2]
    

    elif l2 > l1:
        if pad:
          s1 = np.pad(s1, (0, l2 - l1))
        else:
          s2 = s2[:l1]
    

    return s1, s2



def plot_signals(x, y, e, sr):

    t = np.arange(len(y)) / sr

    _, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)


    ax[0].plot(t, x, label='Original Signal')
    ax[0].set_title('Original Signal')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(t, y, label='Echoed Signal', color='red')
    ax[1].set_title('Echoed Signal')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()
    ax[1].grid(True) 


    ax[2].plot(t, e, label='Filtered Signal', color='orange')
    ax[2].set_title('Filtered Signal')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Amplitude')
    ax[2].legend()
    ax[2].grid(True) 

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
  main()
  