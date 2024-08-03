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

    def filter_time_domain(self, input_data=None, desired_data=None, mu=None, filter_order=None):
        
        if input_data == None:
            input_data = self.input_audio_data

        if desired_data == None:
            desired_data = self.desired_audio_data

        if mu == None:
            self.mu = mu = 0.15

        if filter_order == None:
            self.filter_order = filter_order = 256    #just for plotting



        num_samples = len(input_data)
        weights = np.zeros(self.filter_order)
        p = self.filter_order
        y = np.zeros(num_samples)
        e = np.zeros(num_samples)

        # https://en.wikipedia.org/wiki/Least_mean_squares_filter#LMS_algorithm_summary
        epochs = 1      #epochs Don't seem to have an effect
        for i in range(epochs):
            print("Epoch #: " + str(i+1))
            e = np.zeros(num_samples)
            weights = np.zeros(self.filter_order)

            for n in range(0, num_samples - p):
                x = input_data[n + p:n:-1]
                y[n] = np.dot(weights, x)
                e[n] = desired_data[n] - y[n]

                weights = weights + mu * e[n] * x

            # #this prevents the time shifting
            # for n in range(p, num_samples): #start at filter order to avoid oob error with array
            #     x = input_data[n-p:n]
            #     y[n] = np.dot(weights, x)
            #     e[n] = desired_data[n] - y[n]

            #     # #to prevent errors with audio clipping if weights are too large
            #     # #if filtered sounds weird, then reduce mu or filter order
            #     # weight_update = 2 * mu * e[n] * x
            #     # if np.any(np.abs(weight_update) > 1):
            #     #     weight_update = weight_update / np.max(np.abs(weight_update))

            #     weights = weights + mu * e[n] * x

        self.done_time = inout.get_datetime_string()
        print("Done")

        output_file = self.folder + r"/output/" + self.done_time + "_time_filtered" + ".mp3"
        inout.save_audio(output_file, y, self.input_sr)

        return e, y, weights
    
    def get_metrics(self):
        pass

if __name__ == "__main__":
    print("Creating Echo Signal...")

    # samples_folder = inout.get_samples_folder_path()

    # filename = samples_folder + r"/female.wav"
    # x, sr  = librosa.load(filename,sr=8000)
    # filename = samples_folder + r"/male.wav"
    # d, sr  = librosa.load(filename,sr=8000)

    # rt60_tgt = 1.5   # desired reverberation time (the time in seconds for the energy to drop by 60 dB)
    # room_dim = [5, 5, 5]

    # e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    # room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
    # room.add_source([1.5, 1.5, 1.5])
    # room.add_microphone([0.1, 0.5, 0.1])
    # room.compute_rir()
    # rir = room.rir[0][0]  #rir = room impulse response
    # rir = rir[np.argmax(rir):]  

    # y = np.convolve(x,rir)  #x[n] * h[n] = echo component of final signal
    # scale = np.sqrt(np.mean(x**2)) /  np.sqrt(np.mean(y**2))
    # y = y*scale

    # L = max(len(y),len(d))
    # y = np.pad(y,[0,L-len(y)])
    # d = np.pad(d,[L-len(d),0])
    # x = np.pad(x,[0,L-len(x)])

    # output_folder = inout.get_output_folder_path()
    # filename = output_folder + r"/original_signal.wav"
    # inout.save_audio(filename, x, sr)
    # filename = output_folder + r"/echoed_signal.wav"
    # inout.save_audio(filename, y, sr)

    # lms = LMS(y, x, sr)
    lms = LMS()
    error, output, weights = lms.filter_time_domain()

    #error is the difference between desired and filtered
    #output is the filtered
    #weights are just the weights used

    t = np.arange(len(output)) / lms.input_sr

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.title('Desired Signal')
    plt.plot(t, lms.desired_audio_data)

    plt.subplot(4, 1, 2)
    plt.title('Echo Signal')
    plt.plot(t, lms.input_audio_data)

    plt.subplot(4, 1, 3)
    plt.title('Filtered Signal')
    plt.plot(t, output)

    plt.subplot(4, 1, 4)
    plt.title('Error Signal (Echo Cancelled)')
    plt.plot(t, error)

    plt.suptitle(f'Filter Order: {lms.filter_order}, Mu: {lms.mu}', fontsize=14)
    plt.tight_layout()
    output_file = inout.get_output_folder_path() + lms.done_time + "_graph.png"
    plt.savefig(output_file)
    plt.show()