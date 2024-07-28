import input_output as inout
import numpy as np
from matplotlib import pyplot as plt

class LMS:
    def __init__(self):
        self.folder = inout.get_current_folder()

        mp3_file = self.folder + r"/samples/CUT1-ECHO.mp3"
        self.input_audio_data, self.input_sr = inout.read_audio(mp3_file)

        mp3_file = self.folder + r"/samples/CUT1-ORIG.mp3"
        self.desired_audio_data, self.desired_sr = inout.read_audio(mp3_file)

        #these are the same length mp3s, but for future..
        min_length = min(len(self.desired_audio_data), len(self.input_audio_data))
        self.input_audio_data[:min_length]
        self.desired_audio_data[:min_length]

        self.mu = None
        self.filter_order = None


    def filter_freq_domain(self, input_data=None, desired_data=None, mu=None, filter_order=None):
        pass


    def filter_time_domain(self, input_data=None, desired_data=None, mu=None, filter_order=None):
        
        if input_data == None:
            input_data = self.input_audio_data

        if desired_data == None:
            desired_data = self.desired_audio_data

        if mu == None:
            self.mu = 0.001
            mu = 0.001

        if filter_order == None:
            self.filter_order = 4096    #just for plotting
            filter_order = 4096


        print("Data Length: " + str(len(input_data)) + "\tmu: " + str(mu) + "\tfilter order: " + str(filter_order))

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
                e[n] = desired_data[n + p] - y[n]

                weights = weights + mu * e[n] * x

                # #to prevent errors with audio clipping if weights are too large
                # #if filtered sounds weird, then reduce mu or filter order
                # weight_update = 2 * mu * e[n] * x
                # if np.any(np.abs(weight_update) > 1):
                #     weight_update = weight_update / np.max(np.abs(weight_update))

                # weights += weight_update

        self.done_time = inout.get_datetime_string()
        print("Done")

        output_file = self.folder + r"/output/" + self.done_time + "_time_filtered" + ".mp3"
        inout.save_audio(output_file, y, self.input_sr)

        return e, y, weights

    def get_output_folder(self):
        return self.folder + r"/output/"


if __name__ == "__main__":
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
    output_file = lms.folder + r"/output/" + lms.done_time + "_graph.png"
    plt.savefig(output_file)
    plt.show()