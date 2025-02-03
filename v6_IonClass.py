import math
import numpy as np
import copy as cpy
import scipy.stats as sp
from scipy import signal as sig
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import traceback


class Ion:
    def __init__(self):
        # Overall aggregate trace
        self.trace_frequency = []
        self.trace_amplitude = []
        self.harmonic_trace_amplitude = []
        self.paired = 0

    def reset_pairings(self):
        self.paired = 0


def ion_sort_key(ion):
    return np.average(ion.trace_frequency[0])


class IonField:
    def __init__(self, min_trace_length, max_traceable_jump_neg, max_traceable_jump_pos, max_amp_change, resolution,
                 f_offset, step_length,
                 window_length):
        self.max_traceable_jump_neg = max_traceable_jump_neg
        self.max_traceable_jump_pos = max_traceable_jump_pos
        self.min_trace_length = min_trace_length
        self.max_amp_change = max_amp_change
        self.ions = []
        self.dead_ions = []
        self.ions_wrote = []
        self.difference_matrix = []
        self.amp_difference_matrix = []
        self.Zxx = []

        # For plotting nice graphs:
        self.resolution = resolution
        self.f_offset = f_offset
        self.step_length = step_length
        self.window_length = window_length

    def reset_ion_pairs(self):
        for ion in self.ions:
            ion.paired = 0

    def update_diff_matrix(self, new_ions):
        numberOfRows = len(self.ions)
        numberOfCols = len(new_ions)
        self.difference_matrix = np.empty([numberOfRows, numberOfCols])
        self.amp_difference_matrix = np.empty([numberOfRows, numberOfCols])
        for row in range(numberOfRows):
            for col in range(numberOfCols):
                # Rows correspond to existing ions
                # Cols correspond to newly identified peaks
                newCenter = new_ions[col].trace_frequency[0]
                newAmp = new_ions[col].trace_amplitude[0]
                oldCenter = self.ions[row].trace_frequency[-1]
                oldAmp = self.ions[row].trace_amplitude[-1]
                self.difference_matrix[row, col] = newCenter - oldCenter
                self.amp_difference_matrix[row, col] = newAmp - oldAmp

    def ion_hunter(self, STFT_slice, min_height, min_separation):
        self.Zxx.append(STFT_slice)
        new_ions = []

        # Only search for peaks in the bottom half of the STFT slice
        max_searchable_index = int(
            (((len(STFT_slice) * self.resolution + self.f_offset) / 2) - self.f_offset) / self.resolution)
        peak_indices, properties = sig.find_peaks(STFT_slice[0:max_searchable_index], height=min_height,
                                                  distance=min_separation)
        peak_indices = peak_indices.tolist()

        for peakIndex in peak_indices:
            harm_index = int((((peakIndex * self.resolution + self.f_offset) * 2) - self.f_offset) / self.resolution)
            newIon = Ion()
            newIon.trace_frequency.append(peakIndex)
            newIon.trace_amplitude.append(STFT_slice[peakIndex])
            newIon.harmonic_trace_amplitude.append(STFT_slice[harm_index])
            new_ions.append(newIon)

        if len(self.ions) != 0 and len(new_ions) != 0:
            self.update_diff_matrix(new_ions)

            # Sort diff matrix rows in order of minimum values... should pair 'easy' ions off first
            col_of_mins = np.argmin(abs(self.difference_matrix), axis=1)
            flattened_mins = np.zeros(len(self.ions))
            for i in range(len(self.ions)):
                flattened_mins[i] = abs(self.difference_matrix[i, col_of_mins[i]])

            row_sorted_indices = np.argsort(flattened_mins)
            self.difference_matrix = self.difference_matrix[row_sorted_indices, :]

            ionsCopy = cpy.deepcopy(self.ions)
            for i in range(len(self.ions)):
                self.ions[i] = ionsCopy[row_sorted_indices[i]]

            # Begin pairing off peaks
            for index in range(len(self.ions)):
                # Find the closest new peak to existing ion
                min_diff_index = np.argmin(abs(self.difference_matrix[index, :]))

                # Merge peak and ion if:
                difference = self.difference_matrix[index, min_diff_index]
                amp_difference = self.amp_difference_matrix[index, min_diff_index]
                # and abs(amp_difference) <= self.max_amp_change:
                if self.max_traceable_jump_neg < difference <= self.max_traceable_jump_pos:
                    # RECALL: ion pairing is INCLUDED in the merge function (for the ION, not the merged peak)
                    self.ions[index].trace_frequency.append(new_ions[min_diff_index].trace_frequency[0])
                    self.ions[index].trace_amplitude.append(new_ions[min_diff_index].trace_amplitude[0])
                    self.ions[index].harmonic_trace_amplitude.append(
                        new_ions[min_diff_index].harmonic_trace_amplitude[0])
                    new_ions[min_diff_index].paired = 1
                    self.ions[index].paired = 1
                    # Arbitrarily large number, should never be picked...
                    self.difference_matrix[:, min_diff_index] = 1000000


            deleteFlag = 1
            while deleteFlag:
                counter = 0
                deleteFlag = 0
                while counter < len(self.ions):
                    if self.ions[counter].paired == 0:
                        if len(self.ions[counter].trace_frequency) >= self.min_trace_length:
                            # Toss ions for garbage collection
                            removed_element = self.ions.pop(counter)
                            self.dead_ions.append(removed_element)
                            deleteFlag = 1
                        else:
                            # Toss ions without garbage collection
                            self.ions.pop(counter)
                            deleteFlag = 1

                    counter = counter + 1
            self.reset_ion_pairs()

        else:
            # Spin up ions traces the first time that a new file is loaded
            self.ions = new_ions
            self.difference_matrix = np.empty([len(new_ions), len(new_ions)])

    def sort_traces(self):
        self.ions.sort(key=ion_sort_key)

    def post_process_traces(self, exclude_freq_list, tolerance):
        # Append the 'dead' ions to the ion list
        for ion in self.dead_ions:
            self.ions.append(ion)

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

        deleteFlag = 1
        while deleteFlag:
            counter = 0
            deleteFlag = 0
            while counter < len(self.ions):
                if len(self.ions[counter].trace_frequency) <= self.min_trace_length:
                    self.ions.pop(counter)
                    deleteFlag = 1
                counter = counter + 1

        # Clean up leftover tracing configs and sort by frequency
        self.reset_ion_pairs()  # NOTE: Using .paired property differently than before (1 is bad, 0 is good)
        self.sort_traces()

    def plot_ion_traces(self):
        reoriented_Zxx = np.flipud(np.rot90(np.array(self.Zxx), k=1))

        for i in range(len(self.ions)):
            try:
                plt.plot(np.array(self.ions[i].trace_frequencies) * self.resolution + self.f_offset)
            except Exception as e:
                print('Trace plotting error:', e)

        plot_steps = len(reoriented_Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
        plot_height = len(reoriented_Zxx)

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[
            slice(self.f_offset, plot_height * self.resolution + self.f_offset, self.resolution),
            slice(0, plot_steps, 1)]
        plt.pcolormesh(x, y, reoriented_Zxx, cmap='hot')
        plt.colorbar()
        plt.show()

    def plot_paired_traces(self):
        plt.subplot(1, 2, 1)

        reoriented_Zxx = np.flipud(np.rot90(np.array(self.Zxx), k=1))

        for i in range(len(self.ions)):
            try:
                plt.plot(np.array(self.ions[i].trace_frequency) * self.resolution + self.f_offset)
            except Exception:
                print("Can't plot traces.")

        plot_steps = len(reoriented_Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
        plot_height = len(reoriented_Zxx)

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[
            slice(self.f_offset, plot_height * self.resolution + self.f_offset, self.resolution),
            slice(0, plot_steps, 1)]
        plt.pcolormesh(x, y, reoriented_Zxx, cmap='hot')
        plt.colorbar()
        plt.title("Total Traces")
        plt.xlabel('Time Segments')
        plt.ylabel('Frequency Bins')

        plt.subplot(1, 2, 2)
        for i in range(len(self.ions)):
            try:
                plt.plot(np.array(self.ions[i].trace_frequency) * self.resolution + self.f_offset, color='green')
                plt.plot((np.array(self.ions[i].trace_frequency) * self.resolution + self.f_offset) * 2,
                         color='green', linestyle='dashdot')
            except Exception:
                print("Can't plot traces w/ harmonics.")

        plt.pcolormesh(x, y, reoriented_Zxx, cmap='hot')
        plt.colorbar()
        plt.title("Paired Traces")
        plt.xlabel('Time Segments')
        plt.ylabel('Frequency Bins')
        plt.show()

    def write_ions_to_files(self, save_file_folder, file, tracesHeader, export_Zxx_files):
        try:
            # Assumes files are named with xxxxx.B.txt convention
            if len(self.ions) > 0:
                source_file_name = file.parts[-1]
                source_file_name = source_file_name.split(".B.txt")[0]
                save_file_folder = save_file_folder / source_file_name
                save_file_folder.mkdir(exist_ok=True, parents=True)
                savedIons = 0
                save_file_name = source_file_name + "_Zxx" + ".Zxx"
                save_file_path_full = save_file_folder / save_file_name
                save_file_folder.touch(save_file_name)
                fileDivider = "//"
                ##################################################################
                if export_Zxx_files:
                    Zxx = np.array2string(np.array(self.Zxx))
                    writeList = [tracesHeader] + [fileDivider] + [Zxx]
                    writeString = ", ".join(str(x) for x in writeList)
                    save_file_path_full.write_text(writeString)
                ##################################################################
                written_ions = []
                tolerance = 24  # In Hurtz
                for counter in range(len(self.ions)):
                    write_ion = True
                    for written_ion in written_ions:
                        if abs(self.ions[counter].trace_frequency[0] * self.resolution + self.f_offset - written_ion * 2) <= tolerance or \
                                abs(self.ions[counter].trace_frequency[0] * self.resolution + self.f_offset - written_ion / 2) <= tolerance:
                            write_ion = False
                            print("Rejected harmonic ion at: " + str(self.ions[counter].trace_frequency[0] * self.resolution + self.f_offset))

                    if write_ion:
                        self.ions_wrote.append(self.ions[counter])
                        save_file_name = source_file_name + "_trace_" + str(counter) + ".tv7f"
                        save_file_path_full = save_file_folder / save_file_name
                        save_file_folder.touch(save_file_name)
                        fileDivider = "//"
                        traceFloat = [float(i) for i in self.ions[counter].trace_frequency]
                        magFloat = [float(i) for i in self.ions[counter].trace_amplitude]
                        traceFloatHarm = [float(i) for i in self.ions[counter].harmonic_trace_amplitude]

                        writeList = [tracesHeader] + [fileDivider] + traceFloat + [fileDivider] + magFloat + \
                                    [fileDivider] + traceFloatHarm
                        writeString = ", ".join(str(x) for x in writeList)
                        savedIons = savedIons + 1
                        written_ions.append(self.ions[counter].trace_frequency[0] * self.resolution + self.f_offset)
                        save_file_path_full.write_text(writeString)
                        print("Ion starting at: " + str(traceFloat[0] * self.resolution + self.f_offset))
                print("Saved " + str(savedIons) + "/" + str(len(self.ions)) + " traces to files!")
        except:
            print("Error saving file...")

    def save_png(self, save_file_folder, file, min_freq, max_freq, harm=False, plot_traces=False, vmax=30):
        Zxx = np.flipud(np.rot90(np.array(self.Zxx)))
        # Assumes files are named with xxxxx.B.txt convention
        try:
            if len(self.ions_wrote) > 0:
                source_file_name = file.parts[-1]
                source_file_name = source_file_name.split(".B.txt")[0]
                save_file_folder = save_file_folder / source_file_name
                save_file_folder.mkdir(exist_ok=True, parents=True)
                if harm:
                    save_file_name = source_file_name + "_Zxx_harm" + ".png"
                else:
                    save_file_name = source_file_name + "_Zxx" + ".png"
                save_file_path_full = save_file_folder / save_file_name

                fig, ax = plt.subplots(layout='tight', figsize=(13, 6.5))
                plot_steps = len(Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
                plot_height = len(Zxx)
                y_vals = range(int(self.f_offset), int(plot_height * self.resolution + self.f_offset),
                                int(self.resolution))

                min_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - min_freq))
                max_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - max_freq))

                # generate 2 2d grids for the x & y bounds
                y, x = np.mgrid[
                    slice(y_vals[min_freq_index], y_vals[max_freq_index], self.resolution),
                    slice(0, plot_steps, 1)]

                # Uncomment to plot traces on top of PNG
                # for i in range(len(self.ions)):
                #     try:
                #         ax.plot(np.array(self.ions[i].trace_frequency) * self.resolution + self.f_offset)
                #     except Exception:
                #         print("Can't plot traces.")

                jacob = ax.pcolormesh(x, y, Zxx[0:-1][min_freq_index:max_freq_index], shading='gouraud', cmap='hot',
                                      vmax=vmax)
                fig.colorbar(jacob)
                ax.set_title("")
                x_label = 'Time (ms)'
                ax.set_xlabel(x_label, fontsize=24, weight='bold')
                ax.set_ylabel('Frequency (Hz)', fontsize=24, weight='bold')
                ax.set_xticks([50, 100, 150], ["250", "500", "750"])
                # ax.set_yticks([14250, 14500, 14750, 15000, 15250])
                # ax.set_yticks([13000, 13250, 13500, 13750])
                ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
                ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
                ax.minorticks_on()
                ax.tick_params(axis='x', which='minor', width=3, length=4)
                ax.tick_params(axis='y', which='minor', width=3, length=4)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_linewidth(3)
                ax.spines['top'].set_linewidth(3)

                if plot_traces:
                    for i in range(len(self.ions_wrote)):
                        try:
                            plt.plot(np.array(self.ions_wrote[i].trace_frequency) * self.resolution + self.f_offset,
                                     color='green', linestyle='dashdot')
                        except Exception:
                            print("Can't plot traces w/ harmonics.")

                plt.savefig(save_file_path_full, bbox_inches='tight', dpi=300.0, pad_inches=0.5)
                print("Saved plot with " + str(len(self.ions_wrote)) + " traces!")
                plt.close('all')
                plt.close('all')
        except:
            print("Error saving png file")
