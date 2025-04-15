import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy import signal
import matplotlib.pyplot as plt
import math

root = tk.Tk()
root.withdraw()

# Read BINARY files from Alazar Card (volts stored as binary)
# Ask for data file (output of filter)
file_paths = filedialog.askopenfilenames()
datavolts = []
for path in file_paths:
    newData = np.fromfile(path, dtype=float)
    if len(datavolts) == 0:
        datavolts = np.zeros(len(newData))

    datavolts = datavolts + newData
datavolts = np.asarray(datavolts)

# Ask for reference file (input to filter)
file_paths = filedialog.askopenfilenames()
referencevolts = []
for path in file_paths:
    newData = np.fromfile(path, dtype=float)
    if len(referencevolts) == 0:
        referencevolts = np.zeros(len(newData))

    referencevolts = referencevolts + newData
referencevolts = np.asarray(referencevolts)

# Read CSV files from Keithley DMM
# datavolts = []
# with open('/Users/mmcpartlan/Desktop/DMM-1 Run 5 2022-02-14T16.25.25.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         if row['DMM-1 Voltage (V)'].replace('.', '', 1).isdigit():
#             datavolts.append(float(row['DMM-1 Voltage (V)']))

# Read CSV files from simulated signals
# datavolts = []
# with open('/Users/mmcpartlan/Desktop/freqTestFile_noTime.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for element in reader.fieldnames:
#         datavolts.append(float(element))

sigDuration = 10  # In seconds
segmentLength = 5  # Segment length in milliseconds
sampleFreq = len(datavolts) / sigDuration
pointsPerSegment = int((segmentLength / 1000) * sampleFreq)
segments = int(len(datavolts) / pointsPerSegment)

VVgain = np.zeros(segments)
EEgain = np.zeros(segments)
absoluteOutput = np.zeros(segments)
absoluteEnergy = np.zeros(segments)
refFreqs = np.zeros(segments)
dataFreqs = np.zeros(segments)

for segment in range(segments):
    # Starts at zero, iterates to segment = segments - 1
    dataChunk = datavolts[segment * pointsPerSegment:(segment + 1) * pointsPerSegment]
    chunk_time = sigDuration / segments
    fs = pointsPerSegment / chunk_time
    Hz_mag_data = np.abs(np.fft.fft(dataChunk))
    Hz_mag_data = Hz_mag_data[:len(Hz_mag_data)//2]
    freqs_data = np.fft.fftfreq(pointsPerSegment, 1 / fs)
    freqs_data = freqs_data[:len(freqs_data) // 2]
    chunkMaxMagData_idx = np.where(abs(Hz_mag_data - max(Hz_mag_data))
                                   == abs(Hz_mag_data - max(Hz_mag_data)).min())[0]
    Hz_data_energy = np.trapz(Hz_mag_data, freqs_data)

    if len(chunkMaxMagData_idx) > 1:
        chunkMaxMagData_idx = chunkMaxMagData_idx[0]
    chunkMaxMagData_idx = int(chunkMaxMagData_idx)

    referenceChunk = referencevolts[segment * pointsPerSegment:(segment + 1) * pointsPerSegment]
    Hz_mag_reference = np.abs(np.fft.fft(referenceChunk))
    Hz_mag_reference = Hz_mag_reference[:len(Hz_mag_reference) // 2]
    freqs_reference = np.fft.fftfreq(pointsPerSegment, 1 / fs)
    freqs_reference = freqs_reference[:len(freqs_reference) // 2]
    chunkMaxMagReference_idx = np.where(abs(Hz_mag_reference - max(Hz_mag_reference))
                                        == abs(Hz_mag_reference - max(Hz_mag_reference)).min())[0]
    Hz_reference_energy = np.trapz(Hz_mag_reference, freqs_reference)

    if len(chunkMaxMagReference_idx) > 1:
        chunkMaxMagReference_idx = chunkMaxMagReference_idx[0]
    chunkMaxMagReference_idx = int(chunkMaxMagReference_idx)

    absoluteEnergy[segment] = Hz_data_energy
    absoluteOutput[segment] = Hz_mag_data[chunkMaxMagData_idx]
    VVgain[segment] = Hz_mag_data[chunkMaxMagData_idx] / Hz_mag_reference[chunkMaxMagReference_idx]
    EEgain[segment] = Hz_data_energy/Hz_reference_energy
    refFreqs[segment] = freqs_reference[chunkMaxMagReference_idx]
    dataFreqs[segment] = freqs_data[chunkMaxMagData_idx]

plt.plot(refFreqs[:len(refFreqs) - int(200/segmentLength)], VVgain[:len(VVgain) - int(200/segmentLength)])
plt.ylabel('|Hjw|')
plt.xlabel('Log(f)')
plt.xscale('log')
plt.show()
