import numpy as np
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog as fd
from scipy import signal as sig
import v6_IonClass as ionTracer
import warnings

warnings.filterwarnings('ignore')


def choose_save_folder():
    win = tk.Tk()
    win.focus_force()
    win.withdraw()
    folder = fd.askdirectory(title="Choose save folder")
    return folder


def choose_top_folders(termString=None):
    if termString is not None:
        folders = []
        win = tk.Tk()
        win.focus_force()
        win.withdraw()
        folder = fd.askdirectory(title="Choose top folder")
        walkResults = os.walk(folder)
        for root, dirs, files in walkResults:
            for dir in dirs:
                if dir.endswith(termString):
                    folders.append(os.path.join(root, dir))
        return folders
    else:
        win = tk.Tk()
        win.focus_force()
        win.withdraw()
        folder = fd.askdirectory(title="Choose top folder")
        return [folder]


def generate_filelist(folders, termString):
    # NOTE: folders variable must be a list, even if it is a list of one
    filelists = []
    for folder in folders:
        filelist = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(termString):
                    filelist.append(os.path.join(root, file))
        filelists.append(filelist)
    return filelists


# WARNING: EXPORTING ZXX FILES FOR BACKGROUND GENERATION IS SLOW AND TAKES A TON OF SPACE
export_Zxx_files = False
export_image_files = True


class AnalConfig:
    def __init__(self, SPAMM=2):
        # Parameters for STFT, indexed for each directory to be analyzed
        # Input range for 16-bit digitizer card
        if SPAMM == 3:
            self.voltage_scale = 0.04
        else:
            self.voltage_scale = 0.2
        self.fs = 1000000  # Sampling frequency in Hz
        self.segment_length = 25  # Segment length in ms
        self.step_length = 5  # How many ms to advance the window
        self.zerofill = 250  # How many ms worth of data to zerofill
        self.max_coexisting_ions = 20
        self.ignore_freq_tolerance = 200
        self.ignore_list_check = [69328, 67564]
        self.ignore_list = [69328, 67564]  # Traces to exclude (known noise peaks)

        self.low_freq = 14000  # Lower bound on region of interest
        self.high_freq = 40000  # Upper bound on region of interest
        self.min_trace_charge = 40  # Minimum amplitude to trace (default 25)
        self.min_trace_length = 25
        self.max_traceable_jump_neg = -5000  # Only used when bipolar jumps are allowed
        self.max_traceable_jump_pos = 50
        self.max_amp_change = 200  # NOT USED CURRENTLY


cfg = AnalConfig(SPAMM=2)

check_start = 100  # Start check at x ms
check_length = 100  # ms length of checking transform
magnitude_to_charge_factor = 682500  # Magic number to convert amplitude to approximate charge
pts_per_seg = cfg.fs * (cfg.segment_length / 1000)  # Points per FT segment
zerofill_pts = cfg.fs * (cfg.zerofill / 1000)  # Zerofill points added per segment
pts_per_step = cfg.fs * (cfg.step_length / 1000)  # Points to move the window ahead by
f_reso = cfg.fs / zerofill_pts  # Zero-filled resolution, not true resolution
cfg.max_traceable_jump_neg = cfg.max_traceable_jump_neg / f_reso  # Convert Hertz given into bin space
cfg.max_traceable_jump_pos = cfg.max_traceable_jump_pos / f_reso  # Convert Hertz given into bin space
min_trace_spacing = 4 * f_reso  # Hz (allowed between peaks in checkFFT and STFT without culling the smaller one)
pts_per_check = cfg.fs * (check_length / 1000)  # Points in the spot check section
f_reso_check = cfg.fs / pts_per_check
low_freq_pts = int(cfg.low_freq / f_reso)
high_freq_pts = int(cfg.high_freq / f_reso)
harm_low_freq = cfg.low_freq * 2
harm_high_freq = cfg.high_freq * 2
harm_low_freq_pts = int(harm_low_freq / f_reso)
harm_high_freq_pts = int(harm_high_freq / f_reso)
culling_dist_samps = min_trace_spacing / f_reso


def one_file(file, save_dir):
    file = Path(file)
    save_dir = Path(save_dir)
    folder = file.parts[-3].split(".data")[0] + ".tv7"
    trace_save_directory = save_dir / folder
    print(file)

    data_bits = np.fromfile(file, dtype=np.uint16)  # Import file
    data_volts = ((cfg.voltage_scale / 66536) * 2 * data_bits) - cfg.voltage_scale  # Convert from uint16 to volts
    data_volts_nopulse = data_volts[cfg.step_length * int(cfg.fs / 1000) + 702:]  # Cut out trap pulse at beginning
    max_time = (len(data_volts_nopulse) / cfg.fs) * 1000  # Signal length after cutting out pulse

    ################################################################################################################
    # Calculate the check section to estimate ion presence
    ################################################################################################################
    check_part = sig.detrend(data_volts_nopulse[int(cfg.fs * check_start / 1000):
                                                int(cfg.fs * (check_start + check_length) / 1000)], type='constant')

    check_spec = np.fft.fft(check_part, n=int(pts_per_check))
    check_magnitude = np.abs(check_spec) * magnitude_to_charge_factor
    # Only look at freqs of interest. Apply 1/N scaling to amplitude (not in DFT equation)
    magnitude_slice = check_magnitude[
                      int(cfg.low_freq / f_reso_check):int(cfg.high_freq / f_reso_check)] / pts_per_check

    # Pick peaks out, then analyze the whole file if there is anything there
    sep_distance = min_trace_spacing / f_reso_check
    if sep_distance < 1:
        sep_distance = 1

    peak_indexes, check_prop = sig.find_peaks(magnitude_slice[0:int(len(magnitude_slice) / 2)],
                                              height=cfg.min_trace_charge,
                                              distance=sep_distance)
    check_peaks = peak_indexes * f_reso_check + cfg.low_freq
    real_peaks = []
    for peak in check_peaks:
        isreal = True
        for noise in cfg.ignore_list_check:
            if abs(peak - noise) < cfg.ignore_freq_tolerance:
                isreal = False
        if isreal:
            real_peaks.append(peak)

    # print("Resolution: " + str(f_reso))
    print("Check peaks: " + str(check_peaks.size))

    if 0 < len(real_peaks) < cfg.max_coexisting_ions:

        ################################################################################################################
        # Calculate STFT and trace the full file....
        ################################################################################################################
        f, t, Zxx = sig.stft(data_volts_nopulse, window='hamming', detrend='constant',
                             fs=cfg.fs, nperseg=int(pts_per_seg), nfft=int(zerofill_pts),
                             noverlap=int(pts_per_seg - pts_per_step))

        STFT = np.abs(Zxx) * magnitude_to_charge_factor
        STFT_cut = STFT[low_freq_pts:high_freq_pts]
        pick_time_full = np.arange(0, max_time / cfg.step_length)
        pick_time_pts = pick_time_full[int(0.6 + (cfg.segment_length / 2) / cfg.step_length):-int(
            0.6 + (cfg.segment_length / 2) / cfg.step_length)]  # cut one half seg +1 from each edge

        # def __init__(self, min_trace_length, max_traceable_jump, resolution, f_offset, step_length, window_length):
        traces = ionTracer.IonField(cfg.min_trace_length, cfg.max_traceable_jump_neg, cfg.max_traceable_jump_pos,
                                    cfg.max_amp_change, f_reso,
                                    cfg.low_freq, cfg.step_length,
                                    cfg.segment_length)

        ################################################################################################################
        # Ion Tracing / Post-Processing
        ################################################################################################################
        t_range_offset = pick_time_pts[0]
        for t in pick_time_pts:
            # print("Calculating slice: ", t)
            current_slice = STFT_cut[:, int(t)]
            traces.ion_hunter(current_slice, cfg.min_trace_charge, min_trace_spacing)

        traces.post_process_traces(cfg.ignore_list, cfg.ignore_freq_tolerance)


        ################################################################################################################
        # Trace Visualization Calls (for debugging)
        ################################################################################################################
        print("Tracing " + str(len(traces.ions)) + " ions included in plot.")
        # traces.plot_paired_traces()
        tracesHeader = str(cfg.low_freq) + "|" + str(f_reso) + "|" + str(t_range_offset)
        traces.write_ions_to_files(trace_save_directory, file, tracesHeader, export_Zxx_files)
        if export_image_files:
            traces.save_png(trace_save_directory, file, cfg.low_freq, cfg.high_freq / 2, vmax=None, plot_traces=True)


if __name__ == "__main__":
    folders = choose_top_folders()
    file_ending = ".B.txt"  # For SPAMM 2
    # file_ending = ".txt"  # For SPAMM 3
    filelist = generate_filelist(folders, file_ending)
    save_dir = choose_save_folder()
    # cfg.write_cfg_file(save_dir)
    print(save_dir)
    for file in filelist[0]:
        if file.endswith(file_ending):
            one_file(file, save_dir)
