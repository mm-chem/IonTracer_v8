import numpy as np
import csv
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import math

freqStart = 4000
freqEnd = 4000

totalTime = 2.5  # Seconds
pointsPerSecond = 2000000
timeStep = 1/pointsPerSecond
totalSigX = np.array(range(int(totalTime * pointsPerSecond)))
totalSigY = np.zeros(int(totalTime * pointsPerSecond))
totalSigX = totalSigX * timeStep

freqStep = (freqEnd - freqStart)/(totalTime * pointsPerSecond)

index = 0
newSigY = np.zeros(int(totalTime * pointsPerSecond))
for x in totalSigX:
    freq = freqStart + freqStep * index
    totalSigY[index] = math.sin(2 * math.pi * freq * x)
    index = index + 1

# freqs = np.array([100, 500, 1000, 2000, 30000, 300000, 475000])
# totalTime = 1  # Seconds
# pointsPerSecond = 1000000
# timeStep = 1/pointsPerSecond
# totalSigX = np.array(range(totalTime * pointsPerSecond))
# totalSigY = np.zeros(totalTime * pointsPerSecond)
# totalSigX = totalSigX * timeStep
#
# for f in freqs:
#     phase = np.random.rand() * 2 * math.pi
#     print("generating ", f)
#     index = 0
#     newSigY = np.zeros(totalTime * pointsPerSecond)
#     for x in totalSigX:
#         newSigY[index] = math.sin(2 * math.pi * f * x + phase)
#         index = index + 1
#     totalSigY = totalSigY + newSigY

N = len(totalSigY)
fty = fft(totalSigY)[0:N//2]
plt.figure(1)
plt.plot(fftfreq(N, timeStep)[:N//2], np.abs(fty))
plt.show()

plt.figure(1)
phase = np.arctan2(np.imag(fty), np.real(fty))
index = 0

threshold = max(phase) - max(phase) * 0.05
for element in phase:
    if abs(element) < threshold:
        phase[index] = 0
    index = index + 1

plt.plot(fftfreq(N, timeStep)[:N//2], 2.0/N * np.angle(fty))
plt.plot(fftfreq(N, timeStep)[:N//2], 2.0/N * phase)
plt.show()

plt.figure(2)
plt.plot(totalSigX, totalSigY)
plt.show()

print("Writing to file")
with open('/Users/mmcpartlan/Desktop/freqTestFile_wTime.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(totalSigX)
    writer.writerow(totalSigY)

with open('/Users/mmcpartlan/Desktop/freqTestFile_noTime.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(totalSigY)
