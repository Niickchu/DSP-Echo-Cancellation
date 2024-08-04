import input_output as inout
import numpy as np
from matplotlib import pyplot as plt
import librosa
import pyroomacoustics as pra

class LMS:
    def __init__(self, input_data=[], desired_data=[], sr=None):

        self.folder = inout.get_current_folder()

        if not len(input_data):
            mp3_file = self.folder + r"/samples/CUT1-ECHO.mp3"
            self.input_audio_data, self.input_sr = inout.read_audio(mp3_file)
        else:
            self.input_audio_data = input_data
            self.input_sr = sr

        if not len(desired_data):
            mp3_file = self.folder + r"/samples/CUT1-ORIG.mp3"
            self.desired_audio_data, _ = inout.read_audio(mp3_file)
        else:
            self.desired_audio_data = desired_data

        #these are the same length mp3s, but for future..
        min_length = min(len(self.desired_audio_data), len(self.input_audio_data))
        self.input_audio_data[:min_length]
        self.desired_audio_data[:min_length]

        self.mu = None
        self.filter_order = None

    def filter_time_domain(self, mu, filter_order, save_data):
        
        input_data = self.input_audio_data
        desired_data = self.desired_audio_data

        self.mu = mu
        self.filter_order = filter_order    #so I can plot later

        num_samples = len(input_data)
        weights = np.zeros(self.filter_order)
        p = self.filter_order
        y = np.zeros(num_samples)
        e = np.zeros(num_samples)

        # https://en.wikipedia.org/wiki/Least_mean_squares_filter#LMS_algorithm_summary
        printed_flag = False
        for n in range(0, num_samples - p):
            x = input_data[n + p:n:-1]
            y[n] = np.dot(weights, x)
            e[n] = desired_data[n] - y[n]

            # #to prevent errors with audio clipping if weights are too large
            #if filtered sounds weird, then reduce mu or filter order
            weights = weights + mu * e[n] * x
            if np.any(np.abs(weights) > 1):
                if not printed_flag:
                    print("Clipping with Mu: " + str(mu) + " Filter Order: " + str(filter_order))
                    printed_flag = True
                weights = weights / np.max(np.abs(weights))

                # weights = weights + mu * e[n] * x

        #print("Done")

        if save_data:
            self.done_time = inout.get_datetime_string()
            output_file = self.folder + r"/output/" + self.done_time + "_time_filtered" + ".mp3"
            inout.save_audio(output_file, e, self.input_sr)

        return e, y, weights
    

    def compute_mse_and_snr(self, ideal_signal, real_signal):

        ideal_signal = np.asarray(ideal_signal)
        real_signal = np.asarray(real_signal)

        mse = np.mean((ideal_signal - real_signal) ** 2)

        signal_power = np.mean(ideal_signal ** 2)
        noise_power = np.mean((ideal_signal - real_signal) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)

        return mse, snr

    
    def grid_search(self, mu_values, order_values, save_data):

        mse_matrix = np.zeros((len(order_values), len(mu_values)))
        snr_matrix = np.zeros((len(order_values), len(mu_values)))

        best_mse = float('inf')
        best_snr = float('-inf')
        best_mu = None
        best_order = None

        for i, _mu in enumerate(mu_values):
            for j, _order in enumerate(order_values):

                error, _, _ = lms.filter_time_domain(_mu, _order, save_data)

                error = np.clip(error, -1, 1)
                ideal = speech2[0:len(error)]
                mse, snr = self.compute_mse_and_snr(ideal[60000:], error[60000:])

                mse_matrix[j, i] = mse
                snr_matrix[j, i] = snr

                if mse < best_mse:
                    best_mse = mse
                    best_snr = snr
                    best_mu = _mu
                    best_order = _order

        print("Best MSE:", best_mse)
        print("Best SNR(dB):", best_snr)
        print("Best MU:", best_mu)
        print("Best Filter Order:", best_order)

        # https://www.geeksforgeeks.org/how-to-draw-2d-heatmap-using-matplotlib-in-python/
        plt.figure(figsize=(12, 6))
        plt.imshow(mse_matrix, aspect='auto', origin='lower', extent=[0, len(mu_values), 0, len(order_values)])
        plt.colorbar(label='MSE')
        plt.title('MSE Heatmap')
        plt.xlabel('Mu')
        plt.ylabel('Filter Order')
        plt.xticks(ticks=np.arange(len(mu_values)), labels=mu_values)
        plt.yticks(ticks=np.arange(len(order_values)), labels=order_values)
        plt.tight_layout()
        plt.savefig('mse_heatmap.png')

        plt.figure(figsize=(12, 6))
        plt.imshow(snr_matrix, aspect='auto', origin='lower', extent=[0, len(mu_values), 0, len(order_values)])
        plt.colorbar(label='SNR (dB)')
        plt.title('SNR Heatmap')
        plt.xlabel('Mu')
        plt.ylabel('Filter Order')
        plt.xticks(ticks=np.arange(len(mu_values)), labels=mu_values)
        plt.yticks(ticks=np.arange(len(order_values)), labels=order_values)
        plt.tight_layout()
        plt.savefig('snr_heatmap.png')
        
        return best_mse, best_snr, best_mu, best_order

if __name__ == "__main__":
    print("Creating Echo Signal...")

    samples_folder = inout.get_samples_folder_path()

    filename = samples_folder + r"/female.wav"
    speech1, sr  = librosa.load(filename,sr=8000)
    filename = samples_folder + r"/male.wav"
    speech2, sr  = librosa.load(filename,sr=8000)

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

    
    L = max(len(speech1_echoed),len(speech2))
    speech1_echoed = np.pad(speech1_echoed,[0,L-len(speech1_echoed)])
    speech2 = np.pad(speech2,[L-len(speech2),0])
    speech1 = np.pad(speech1,[0,L-len(speech1)])
    near_end_signal = speech2 + speech1_echoed

    output_folder = inout.get_output_folder_path()
    filename = output_folder + r"/echoed_signal.wav"
    inout.save_audio(filename, near_end_signal, sr)

    print("Echo Created")

    lms = LMS(speech1 , near_end_signal, sr)

    mu_vals = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4]
    or_vals = [8, 16, 32, 64, 128]

    print("Starting Grid Search...")
    _, _, best_mu, best_order = lms.grid_search(mu_vals, or_vals, False)
    print("Done")

    error, output, _ = lms.filter_time_domain(best_mu, best_order, True)


    error = np.clip(error,-1,1)
    ideal = speech2[0:len(error)]
    plt.figure(figsize=(12, 8))
            
    t = np.arange(len(speech1)) / sr
    plt.subplot(4, 1, 1)
    plt.title('Far-end signal')
    plt.plot(t, speech1)
    
    t = np.arange(len(near_end_signal)) / sr
    plt.subplot(4, 1, 2)
    plt.title('Microphone picked signal')
    plt.plot(t, near_end_signal, label='Near End Signal')
    plt.plot(t, speech2, color='red', label="Bob's Voice")

    t = np.arange(len(error)) / sr
    plt.subplot(4, 1, 3)
    plt.title('Filtered signal')
    plt.plot(t, error)

    plt.subplot(4, 1, 4)
    plt.title('Ideal signal (if echo did not exist)')
    plt.plot(t, ideal)

    plt.suptitle(f'Filter Order: {lms.filter_order}, Mu: {lms.mu}', fontsize=14)
    plt.tight_layout()
    output_file = inout.get_output_folder_path() + lms.done_time + "_graph.png"
    plt.savefig(output_file)

    mse, snr = lms.compute_mse_and_snr(ideal[60000:], error[60000:])
    plt.show()




