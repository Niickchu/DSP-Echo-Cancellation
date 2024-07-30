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
from matplotlib import pyplot as plt
from frequency_domain_adaptive_filters.fdaf import fdaf


INPUT_DIR = 'samples'
OUTPUT_WAV = 'fdaf_audio'
OUTPUT_PLOTS = 'fdaf_plots'

# M = 256
# MU = 0.1
M_list = [32, 64, 128, 256, 512, 1024, 2048]
MU_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]

def compute_mse_and_snr(ideal_signal, real_signal):
    """
    Compute Mean Squared Error (MSE) and Signal-to-Noise Ratio (SNR) between 
    an ideal theoretical signal and a real practical signal.

    Parameters:
    ideal_signal (np.ndarray): The ideal theoretical signal.
    real_signal (np.ndarray): The real practical signal.

    Returns:
    tuple: A tuple containing MSE and SNR values.
    """
    
    # Ensure both signals are numpy arrays
    ideal_signal = np.asarray(ideal_signal)
    real_signal = np.asarray(real_signal)

    # Calculate MSE
    mse = np.mean((ideal_signal - real_signal) ** 2)

    # Calculate SNR
    signal_power = np.mean(ideal_signal ** 2)
    noise_power = np.mean((ideal_signal - real_signal) ** 2)
    #print(signal_power, noise_power)
    snr = 10 * np.log10(signal_power / noise_power)

    return mse, snr

def main():
    speech1, sr  = librosa.load(f'{INPUT_DIR}/female.wav',sr=8000)
    speech2, _  = librosa.load(f'{INPUT_DIR}/male.wav',sr=8000)

    #reverberation time (the time in seconds for the energy to drop by 60 dB)
    rt60_tgt = 0.08
    room_dim = [2, 2, 2]
    
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source([1.5, 1.5, 1.5])
    room.add_microphone([0.1, 0.5, 0.1])
    room.compute_rir()
    rir = room.rir[0][0]
    rir = rir[np.argmax(rir):]

    speech1_echoed = np.convolve(speech1,rir)
    scale = np.sqrt(np.mean(speech1**2)) /  np.sqrt(np.mean(speech1_echoed**2))
    speech1_echoed = speech1_echoed*scale
    #sf.write(f'{OUTPUT_DIR}/speech1_echoed.wav', speech1_echoed, sr, subtype='PCM_16')
    
    L = max(len(speech1_echoed),len(speech2))
    speech1_echoed = np.pad(speech1_echoed,[0,L-len(speech1_echoed)])
    speech2 = np.pad(speech2,[L-len(speech2),0])
    speech1 = np.pad(speech1,[0,L-len(speech1)])
    near_end_signal = speech2 + speech1_echoed

    #sf.write(f'{OUTPUT_DIR}/speech1.wav', speech1, sr, subtype='PCM_16')
    #sf.write(f'{OUTPUT_DIR}/near_end_signal.wav', near_end_signal, sr, subtype='PCM_16')

    for MU in [0.1]:
        mse_list = []
        snr_list = []
        for M in M_list:
            print(f'Filter Length: {M}, Mu: {MU}')
            e = fdaf(speech1, near_end_signal, M=M, mu=MU)
            e = np.clip(e,-1,1)
            ideal = speech2[0:len(e)]

            # sf.write(f'{OUTPUT_WAV}/M_{M}_mu_{MU}.wav', e, sr, subtype='PCM_16')

            # plt.figure(figsize=(16, 12))
            
            # t = np.arange(len(speech1)) / sr
            # plt.subplot(4, 1, 1)
            # plt.title('Far-end signal')
            # plt.plot(t, speech1)
            
            # t = np.arange(len(near_end_signal)) / sr
            # plt.subplot(4, 1, 2)
            # plt.title('Microphone picked signal')
            # plt.plot(t, near_end_signal)

            # t = np.arange(len(e)) / sr
            # plt.subplot(4, 1, 3)
            # plt.title('Filtered signal')
            # plt.plot(t, e)

            # plt.subplot(4, 1, 4)
            # plt.title('Ideal signal (if echo did not exist)')
            # plt.plot(t, ideal)

            # plt.suptitle(f'Filter Length: {M}, Mu: {MU}', fontsize=10)
            # plt.savefig(f"{OUTPUT_PLOTS}/M_{M}_step_{MU}.png")
            
            mse, snr = compute_mse_and_snr(ideal[60000:], e[60000:])

            #print(f"MSE: {mse}")
            #print(f"SNR: {snr} dB")
            snr_list.append(snr)
            mse_list.append(mse)
        
        plt.figure(figsize=(20, 16))
            
        plt.subplot(2, 1, 1)
        plt.title('Mean Squared Error')
        plt.ylabel('MSE')
        plt.xlabel('Filter Length')
        plt.plot(M_list, mse_list)
        
        plt.subplot(2, 1, 2)
        plt.title('Signal-to-Noise Ratio')
        plt.ylabel('SNR (dB)')
        plt.xlabel('Filter Length')
        plt.plot(M_list, snr_list)

        plt.show()


if __name__ == '__main__':
    main()
    