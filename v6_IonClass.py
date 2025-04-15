import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt


class Ion:
    def __init__(self):
        # Overall aggregate trace
        self.STFT_slice_number = None
        self.trace_frequency = None
        self.trace_amplitude = None


class IonField:
    def __init__(self, resolution, f_offset, step_length, window_length):
        self.ions = []
        self.Zxx = []

        # For plotting nice graphs:
        self.resolution = resolution
        self.f_offset = f_offset
        self.step_length = step_length
        self.window_length = window_length

    def ion_hunter(self, STFT_slice, min_height, min_separation, STFT_slice_number):
        self.Zxx.append(STFT_slice)
        peak_indices, properties = sig.find_peaks(STFT_slice, height=min_height, distance=min_separation)
        peak_indices = peak_indices.tolist()

        for peakIndex in peak_indices:
            newIon = Ion()
            newIon.trace_slice = STFT_slice_number
            newIon.trace_frequency = peakIndex
            newIon.trace_amplitude = STFT_slice[peakIndex]
            self.ions.append(newIon)

    def post_process_traces(self, exclude_freq_list, tolerance):
        deleteFlag = 1
        while deleteFlag:
            counter = 0
            deleteFlag = 0
            while counter < len(self.ions):
                for freq in exclude_freq_list:
                    difference_tolerance = abs(
                        (self.ions[counter].trace_frequency[0] * self.resolution + self.f_offset) - freq) / tolerance
                    if difference_tolerance <= 1:
                        # Toss ions for garbage collection
                        self.ions.pop(counter)
                        deleteFlag = 1
                        break
                counter = counter + 1

    def plot_ion_field(self):
        reoriented_Zxx = np.flipud(np.rot90(np.array(self.Zxx), k=1))

        # for i in range(len(self.ions)):
        #     try:
        #         plt.axvline(np.array(self.ions[i].trace_frequencies) * self.resolution + self.f_offset)
        #     except Exception as e:
        #         print('Trace plotting error:', e)

        plot_steps = len(reoriented_Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
        plot_height = len(reoriented_Zxx)

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[
            slice(self.f_offset, plot_height * self.resolution + self.f_offset, self.resolution),
            slice(0, plot_steps, 1)]
        plt.pcolormesh(x, y, reoriented_Zxx, cmap='hot')
        plt.colorbar()
        plt.show()

