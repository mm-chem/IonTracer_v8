import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog as fd
from scipy import signal as sig
import v6_IonClass as ionTracer
import v1_FilterSim as sigGen
import warnings
from sklearn import preprocessing

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
    def __init__(self):
        # Parameters for STFT, indexed for each directory to be analyzed
        # Input range for 16-bit digitizer card
        self.ignore_freq_tolerance = 100
        self.ignore_list = []
        self.voltage_scale = 0.2
        self.fs = 1000000  # Sampling frequency in Hz
        self.segment_length = 25  # Segment length in ms
        self.step_length = 5  # How many ms to advance the window
        self.zerofill = 250  # How many ms worth of data to zerofill

        self.low_freq = 14000  # Lower bound on region of interest
        self.high_freq = 40000  # Upper bound on region of interest
        self.charge_threshold = 200  # Minimum amplitude to trace (default 25)


cfg = AnalConfig()

magnitude_to_charge_factor = 682500  # Magic number to convert amplitude to approximate charge
pts_per_seg = cfg.fs * (cfg.segment_length / 1000)  # Points per FT segment
zerofill_pts = cfg.fs * (cfg.zerofill / 1000)  # Zerofill points added per segment
pts_per_step = cfg.fs * (cfg.step_length / 1000)  # Points to move the window ahead by
f_reso = cfg.fs / zerofill_pts  # Zero-filled resolution, not true resolution
min_trace_spacing = 4 * f_reso  # Hz (allowed between peaks in checkFFT and STFT without culling the smaller one)


low_freq_pts = int(cfg.low_freq / f_reso)
high_freq_pts = int(cfg.high_freq / f_reso)

def one_file(file):
    file = Path(file)
    print(file)

    data_bits = np.fromfile(file, dtype=np.uint16)  # Import file
    data_volts = ((cfg.voltage_scale / 66536) * 2 * data_bits) - cfg.voltage_scale  # Convert from uint16 to volts
    max_time = (len(data_volts) / cfg.fs) * 1000  # Signal length, in time domain

    ################################################################################################################
    # Lets look at the time domain signal...
    ################################################################################################################
    # Generate square wave
    freqSquare = 14000  # Frequency given in Hz
    simulationTime = (1/freqSquare) * 12 # Estimated time it takes for a 14kHz ion to transit all 24 tubes
    points = cfg.fs * simulationTime
    t = np.linspace(0, simulationTime, int(points))  # Set up sample of 1 second
    squareTest = sig.square(2 * np.pi * freqSquare * t, duty=0.5)
    zero_buffer = np.zeros(len(squareTest))
    final_test = np.concatenate((zero_buffer, squareTest))
    end_zero_buffer = np.zeros(len(data_volts) - len(final_test))
    simulationTime = 1
    final_test = np.concatenate((final_test, end_zero_buffer))

    freqs, timeDomainOutAnalog = sigGen.simulateWfn(sigGen.analogDevicesBPF, final_test, simulationTime)
    plt.plot(np.ndarray.tolist(sigGen.normalized(timeDomainOutAnalog))[0])
    plt.plot(np.ndarray.tolist(sigGen.normalized(final_test))[0])
    plt.show()

    preprocessing.MinMaxScaler()

    correlation = np.correlate()
    plt.plot(correlation)
    # plt.plot(data_volts)
    plt.show()

    # Calculate the moving average
    window_size = 100
    y_smooth = np.convolve(data_volts, np.ones(window_size) / window_size, mode='same')

    # Detrend the data by subtracting the moving average
    data_volts_detrended = data_volts - y_smooth

    # Plot the original and detrended data
    plt.figure(figsize=(10, 5))
    plt.plot(data_volts, label='Original Data')
    plt.plot(data_volts_detrended, label='Detrended Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Detrending with Moving Average')
    plt.legend()
    plt.show()


    ################################################################################################################
    # Calculate STFT just for fun...
    ################################################################################################################
    f, t, Zxx = sig.stft(data_volts, window='hamming', detrend='constant',
                         fs=cfg.fs, nperseg=int(pts_per_seg), nfft=int(zerofill_pts),
                         noverlap=int(pts_per_seg - pts_per_step))

    STFT = np.abs(Zxx) * magnitude_to_charge_factor
    STFT_cut = STFT[low_freq_pts:high_freq_pts]
    pick_time_full = np.arange(0, max_time / cfg.step_length)
    pick_time_pts = pick_time_full[int(0.6 + (cfg.segment_length / 2) / cfg.step_length):-int(
        0.6 + (cfg.segment_length / 2) / cfg.step_length)]  # cut one half seg +1 from each edge

    # def __init__(self, min_trace_length, max_traceable_jump, resolution, f_offset, step_length, window_length):
    traces = ionTracer.IonField(f_reso, cfg.low_freq, cfg.step_length, cfg.segment_length)

    ################################################################################################################
    # Ion Tracing / Post-Processing
    ################################################################################################################
    for t in pick_time_pts:
        current_slice = STFT_cut[:, int(t)]
        traces.ion_hunter(current_slice, cfg.charge_threshold, min_trace_spacing, int(t))
    traces.post_process_traces(cfg.ignore_list, cfg.ignore_freq_tolerance)
    traces.plot_ion_field()
    print(traces.ions)


if __name__ == "__main__":
    # folders = choose_top_folders()
    # file_ending = ".A.bin"  # For SPEED
    # filelist = generate_filelist(folders, file_ending)
    # for file in filelist[0]:
    #     if file.endswith(file_ending):
    #         one_file(file)

    # Set static path for quicker testing...
    one_file("/Users/mmcpartlan/Desktop/SPEeD/SPEeD_Test.data/15_01_08.808.A.bin")