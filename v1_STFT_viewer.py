import numpy as np
import matplotlib.pyplot as plt


class ZxxBackdrop:
    def __init__(self, trace_folder, Zxx_file):
        self.Zxx = None
        self.f_range_offset = None
        self.t_range_offset = None
        self.resolution = None
        self.Zxx_filename = Zxx_file[0][0]
        self.daughter_traces = []
        self.trace_folder = trace_folder
        self.load_Zxx()

    def load_Zxx(self):
        try:
            with open(self.Zxx_filename, newline='') as file:
                Zxx_string = file.read().replace('\n', '')
            Zxx_string = Zxx_string.split("//, ")
            header_data = Zxx_string[0]
            header_data = header_data.split(', ')[0]
            header_data = header_data.split('|')
            self.f_range_offset = float(header_data[0])
            self.resolution = float(header_data[1])
            self.t_range_offset = float(header_data[2])
            Zxx_string = Zxx_string[1]
            Zxx_string = Zxx_string.replace('[', ',')
            Zxx_string = Zxx_string.replace(']', ',')
            Zxx_string = Zxx_string.split(',')
            Zxx_array = []
            for element in Zxx_string:
                if len(element) > 2:
                    element = element.replace(' ', ', ')
                    element = element.replace(', , ', ', ')
                    element = element.split(',')
                    Zxx_array.append([float(i) for i in element])
            self.Zxx = np.flipud(np.rot90(np.array(Zxx_array)))
        except Exception as e:
            print("No ZxxBackdrop detected.", e)

    def find_closest(self, A, target):
        # A must be sorted
        A = (np.array(A))
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A) - 1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx

    def plot_vertical_timeslices(self, start_slice, end_slice, x_start, x_end):
        zxx_cutout = self.Zxx[:, start_slice:end_slice]
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.inferno(np.linspace(0, 1, end_slice -
                                                                                         start_slice)))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        time_step = start_slice
        for column in zip(*zxx_cutout):
            column = np.array(column)
            x_axis = []
            for i in range(len(column)):
                x_axis.append(i * self.resolution + self.f_range_offset)
            x_start_index = self.find_closest(x_axis, x_start)
            x_end_index = self.find_closest(x_axis, x_end)
            ax.plot(x_axis[x_start_index:x_end_index], column[x_start_index:x_end_index], zs=time_step, zdir='z')
            # d2_col_obj = ax.fill_between(x_axis, 0.5, column, step='pre', alpha=0.1)
            # ax.add_collection3d(d2_col_obj, zs=time_step, zdir='z')
            time_step = time_step + 1
        ax.set_xlim(x_start, x_end)
        ax.set_zlim(start_slice, end_slice)
        ax.set_xlabel('Freq (Hz)', fontsize=10)
        ax.set_ylabel('Charge', fontsize=10)
        ax.set_zlabel('STFT Slice', fontsize=10)
        ax.view_init(elev=45, azim=0, roll=90)
        plt.show()

    def plot_all_traces_on_Zxx(self, min_freq, max_freq, include_harmonics=False, plot_trace_overlay=False):
        fig, ax = plt.subplots(layout='tight', figsize=(13, 6.5))
        try:
            if plot_trace_overlay:
                if include_harmonics:
                    for trace in self.daughter_traces:
                        if np.max(trace.trace_harm) < max_freq and np.min(trace.trace) > min_freq:
                            ax.plot(np.array(trace.trace_indices), np.array(trace.trace), color='magenta')
                            ax.plot(np.array(trace.trace_indices), np.array(trace.trace_harm), linestyle='dashdot',
                                    color='magenta')
                else:
                    for trace in self.daughter_traces:
                        if max_freq > np.max(trace.trace) and np.min(trace.trace) > min_freq:
                            if len(trace.fragments) == 1:
                                ax.plot(np.array(trace.trace_indices), np.array(trace.trace), color='magenta')
                            else:
                                for frag in trace.fragments:
                                    ax.plot(np.array(frag.trace_indices), np.array(frag.trace))


        except Exception as e:
            print('Trace plotting error:', e)

        plot_steps = len(self.Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
        plot_height = len(self.Zxx)
        y_vals = range(int(self.f_range_offset), int(plot_height * self.resolution + self.f_range_offset),
                       int(self.resolution))

        min_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - min_freq))
        max_freq_index = min(range(len(y_vals)), key=lambda i: abs(y_vals[i] - max_freq))

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[
            slice(y_vals[min_freq_index], y_vals[max_freq_index], self.resolution),
            slice(self.t_range_offset, plot_steps + self.t_range_offset, 1)]
        ax.pcolormesh(x, y, self.Zxx[0:-1][min_freq_index:max_freq_index], cmap='hot')
        ax.set_title("")
        ax.set_xlabel('Time (ms)', fontsize=24, weight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=24, weight='bold')
        ax.set_xticks([50, 100, 150], ["250", "500", "750"])
        ax.set_yticks([13000, 13250, 13500, 13750, 14000])
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
        save_path = "/Users/mmcpartlan/Desktop/"
        plt.savefig(save_path + 'exported_trace_plot.png', bbox_inches='tight', dpi=300.0, pad_inches=0.5,
                    transparent='true')
