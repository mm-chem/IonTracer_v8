from scipy.fftpack import fft, ifft, fftfreq
from scipy import signal
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def FirstOrderLPFGain(Z1, Z2, R2, R3, f):
    w = 2 * np.pi * f
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = 0.00000001
    Z2 = 1 / (1j * w * Z2)
    Z3 = R2
    Z4 = R3
    Hs = ((Z3 + Z4) / Z4) * (Z2 / (Z1 + Z2))
    return Hs, None


def FirstOrderHPFNoGain(Z1, Z2, f):
    w = 2 * np.pi * f
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = 0.00000001
    Z1 = 1 / (1j * w * Z1)
    Hs = Z2 / (Z1 + Z2)
    return Hs, None


def SallenKeyHPF(Z1, Z2, Z3, Z4, f):
    K = 1
    R = Z3
    C = Z1
    m = Z4 / R
    n = Z2 / C
    Q = np.sqrt(m * n) / (m + 1 + m * n * (1 - K))
    w = 2 * np.pi * f
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = 0.00000001
    Z1 = 1 / (1j * w * Z1)
    Z2 = 1 / (1j * w * Z2)
    Hs = (Z3 * Z4) / (Z1 * Z2 + Z3 * (Z1 + Z2) + Z3 * Z4)
    return Hs, Q


def SallenKeyLPF(Z1, Z2, Z3, Z4, f):
    K = 1
    R = Z2
    C = Z4
    m = Z1 / R
    n = Z3 / C
    Q = np.sqrt(m * n) / (m + 1 + m * n * (1 - K))
    w = 2 * np.pi * f
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = 0.00000001
    Z3 = 1 / (1j * w * Z3)
    Z4 = 1 / (1j * w * Z4)
    Hs = (Z3 * Z4) / (Z1 * Z2 + Z3 * (Z1 + Z2) + Z3 * Z4)
    return Hs, Q


def nF(capacitance):
    return capacitance / 1000000000


def pF(capacitance):
    return capacitance / 1000000000000


def kOhm(resistance):
    return resistance * 1000


def ohm(resistance):
    return resistance


def printCap(capacitance):
    returnString = ""
    if capacitance < 0.000000001:
        returnString = returnString + str(round(capacitance * 1000000000000, 2)) + " pF "
    elif capacitance < 0.000001:
        returnString = returnString + str(round(capacitance * 1000000000, 2)) + " nF "
    elif capacitance < 0.001:
        returnString = returnString + str(round(capacitance * 1000000, 2)) + " uF "
    elif capacitance < 1:
        returnString = returnString + str(round(capacitance * 1000, 2)) + " mF "
    else:
        returnString = returnString + str(round(capacitance, 2)) + " F "
    return returnString


def printRes(resistance):
    returnString = ""
    if resistance < 1000:
        returnString = returnString + str(round(resistance, 2)) + " Ohms "
    elif resistance < 1000000:
        returnString = returnString + str(round(resistance / 1000, 2)) + " kOhms "
    else:
        returnString = returnString + str(round(resistance / 1000000, 2)) + " MOhms "
    return returnString


def getFunctionalPassband(filterToApply, f_min=100, f_max=1000000, cutoff=0.95):
    f_fft = np.logspace(np.log10(f_min), np.log10(f_max), num=1000000)
    Hs = filterToApply(f_fft)
    maxGain = max(abs(Hs))
    cutoffAmplitude = cutoff * maxGain
    differenceArray = abs(Hs) - cutoffAmplitude
    zeroCrossings = np.round(f_fft[np.where(np.diff(np.sign(differenceArray)))[0]])
    # This returns zeroCrossings in Hz
    return zeroCrossings.astype(int)


def getFunctionalRipple(filterToApply, f_min=100, f_max=1000000, cutoff=0.95):
    f_fft = np.logspace(np.log10(f_min), np.log10(f_max), num=1000000)
    Hs = filterToApply(f_fft)
    passband_Hz = getFunctionalPassband(filterToApply, f_min, f_max, cutoff)

    # Calculate the difference array to find nearest point
    difference_array = np.absolute(f_fft - passband_Hz[0])
    low_f_index = difference_array.argmin()
    difference_array = np.absolute(f_fft - passband_Hz[1])
    high_f_index = difference_array.argmin()

    passband = Hs[low_f_index:high_f_index]
    min_Hs = (np.abs(passband)).min()
    max_Hs = (np.abs(passband)).max()
    return max_Hs - min_Hs

def capGetOTC(ideal_cap, OTC_tolerance):
    capacitor_OTC = [pF(10), pF(15), pF(22), pF(33), pF(47), pF(68), pF(100), pF(220), pF(470), pF(680),
                     nF(1), nF(1.5), nF(2.2), nF(3.3), nF(4.7), nF(6.8), nF(10), nF(22), nF(47), nF(68)]
    best_fit = min(capacitor_OTC, key=lambda x: abs(x - ideal_cap))
    if abs(best_fit - ideal_cap)/ideal_cap >= OTC_tolerance:
        print(printCap(ideal_cap))
        return ideal_cap
    else:
        print(printCap(best_fit))
        return best_fit

def resGetOTC(ideal_res, OTC_tolerance):
    resistor_OTC = [ohm(1), ohm(10), ohm(100), kOhm(1), kOhm(10), kOhm(100), kOhm(1000), kOhm(10000),
                    ohm(120), kOhm(1.2), kOhm(12),
                    ohm(1.5), ohm(15), ohm(150), kOhm(1.5), kOhm(15), kOhm(150), kOhm(1500),
                    ohm(180), kOhm(1.8), kOhm(18),
                    ohm(2.2), ohm(22), ohm(220), kOhm(2.2), kOhm(22), kOhm(220), kOhm(2200),
                    ohm(270), kOhm(2.7), kOhm(27),
                    ohm(3.3), ohm(33), ohm(330), kOhm(3.3), kOhm(33), kOhm(330), kOhm(3300),
                    ohm(390), kOhm(3.9), kOhm(39),
                    ohm(4.7), ohm(47), ohm(470), kOhm(4.7), kOhm(47), kOhm(470), kOhm(4700),
                    ohm(560), kOhm(5.6), kOhm(56),
                    ohm(6.8), ohm(68), ohm(680), kOhm(6.8), kOhm(68), kOhm(680), kOhm(6800),
                    ohm(820), kOhm(8.2), kOhm(82)]
    best_fit = min(resistor_OTC, key=lambda x: abs(x - ideal_res))
    if abs(best_fit - ideal_res) / ideal_res >= OTC_tolerance:
        print(printRes(ideal_res))
        return ideal_res
    else:
        print(printRes(best_fit))
        return best_fit

def practicalBPF(f_array, round_to_OTC=True, OTC_tolerance_res=0.03, OTC_tolerance_cap=0.05):
    if round_to_OTC:
        stage1, Q1 = FirstOrderLPFGain(resGetOTC(ohm(768), OTC_tolerance_res), capGetOTC(nF(1), OTC_tolerance_cap), resGetOTC(ohm(470), OTC_tolerance_res), resGetOTC(ohm(47), OTC_tolerance_res), f_array)
        stage2, Q2 = SallenKeyHPF(capGetOTC(nF(20), OTC_tolerance_cap), capGetOTC(nF(20), OTC_tolerance_cap), resGetOTC(ohm(845), OTC_tolerance_res), resGetOTC(kOhm(1.27), OTC_tolerance_res), f_array)
        stage3, Q3 = SallenKeyLPF(resGetOTC(ohm(681), OTC_tolerance_res), resGetOTC(kOhm(1.54), OTC_tolerance_res), capGetOTC(nF(1), OTC_tolerance_cap), capGetOTC(pF(560), OTC_tolerance_cap), f_array)
        stage4, Q4 = FirstOrderHPFNoGain(capGetOTC(nF(20), OTC_tolerance_cap), resGetOTC(kOhm(1.02), OTC_tolerance_res), f_array)
        stage5, Q5 = SallenKeyLPF(resGetOTC(kOhm(1.82), OTC_tolerance_res), resGetOTC(kOhm(4.02), OTC_tolerance_res), capGetOTC(nF(1), OTC_tolerance_cap), capGetOTC(pF(82), OTC_tolerance_cap), f_array)
        stage6, Q6 = SallenKeyHPF(capGetOTC(nF(20), OTC_tolerance_cap), capGetOTC(nF(20), OTC_tolerance_cap), resGetOTC(ohm(316), OTC_tolerance_res), resGetOTC(kOhm(3.32), OTC_tolerance_res), f_array)
    else:
        stage1, Q1 = FirstOrderLPFGain(ohm(768), nF(1), ohm(470), ohm(47), f_array)
        stage2, Q2 = SallenKeyHPF(nF(20), nF(20), ohm(845), kOhm(1.27), f_array)
        stage3, Q3 = SallenKeyLPF(ohm(681), kOhm(1.54), nF(1), pF(560), f_array)
        stage4, Q4 = FirstOrderHPFNoGain(nF(20), kOhm(1.02), f_array)
        stage5, Q5 = SallenKeyLPF(kOhm(1.82), kOhm(4.02), nF(1), pF(82), f_array)
        stage6, Q6 = SallenKeyHPF(nF(20), nF(20), ohm(316), kOhm(3.32), f_array)
    # print("Practical: ", Q1, Q2, Q3, Q4, Q5, Q6)
    return stage1 * stage2 * stage3 * stage4 * stage5 * stage6


def analogDevicesBPF(f_array):
    # Filter parameters specified by Analog Devices Wizard
    stage1, Q1 = FirstOrderLPFGain(ohm(768), nF(1), ohm(470), ohm(47), f_array)
    stage2, Q2 = SallenKeyHPF(nF(20), nF(20), ohm(845), kOhm(1.27), f_array)
    stage3, Q3 = SallenKeyLPF(ohm(681), kOhm(1.54), nF(1), pF(560), f_array)
    stage4, Q4 = FirstOrderHPFNoGain(nF(20), kOhm(1.02), f_array)
    stage5, Q5 = SallenKeyLPF(kOhm(1.82), kOhm(4.02), nF(1), pF(82), f_array)
    stage6, Q6 = SallenKeyHPF(nF(20), nF(20), ohm(316), kOhm(3.32), f_array)
    # print("Theoretical: ", Q1, Q2, Q3, Q4, Q5, Q6)
    return stage1 * stage2 * stage3 * stage4 * stage5 * stage6


def idealBPF(filterToCopy, f):
    w = 2 * np.pi * f
    zeroCrossings = getFunctionalPassband(filterToCopy, f)
    lowCutoff = zeroCrossings[0] * 2 * np.pi
    highCutoff = zeroCrossings[1] * 2 * np.pi
    Hs = np.zeros(len(w))
    for i in range(len(w)):
        if lowCutoff <= w[i] <= highCutoff:
            Hs[i] = 1
    return Hs


def generateFilterResponse(filterToApply, testWave, duration):
    N = len(testWave)
    T = duration
    DT = T / N
    f_fft = fftfreq(N, DT)
    Hs = filterToApply(f_fft)
    Hs = Hs[0:int(len(Hs) / 2)]
    f_fft = f_fft[0:int(len(f_fft) / 2)]
    return f_fft, Hs

def generateFilterResponse_v2(filterToApply, f_min=100, f_max=1000000):
    f_fft = np.logspace(np.log10(f_min), np.log10(f_max), num=1000000)
    Hs = filterToApply(f_fft)
    return f_fft, abs(Hs)


def simulateWfn(filterToApply, testWave, duration, compute_ideal=False, copy_filter=None):
    N = len(testWave)
    T = duration
    DT = T / N
    f_fft = fftfreq(N, DT)
    y_sq_fft = fft(testWave)
    # Apply the user-defined filter (or the ideal filter, if provided
    if not compute_ideal and copy_filter is None:
        y_sq_fft_out = y_sq_fft * filterToApply(f_fft)
    else:
        y_sq_fft_out = y_sq_fft * filterToApply(copy_filter, f_fft)
    y_sq_out = ifft(y_sq_fft_out)
    return f_fft, y_sq_out


if __name__ == '__main__':
    # ========================================================
    # GENERATING A GENERIC SQUARE WAVE INPUT
    # ========================================================
    simulationTime = 0.0005
    points = 5000000
    t = np.linspace(0, simulationTime, int(points))  # Set up sample of 1 second
    freqSquare = 55000  # Frequency given in Hz
    squareTest = signal.square(2 * np.pi * freqSquare * t, duty=0.5)

    # ========================================================
    # CALCULATE STATS FOR THE PRACTICAL BPF (PROPOSED VALUE DESIGN)
    # ========================================================
    print("PROPOSED VALUE DESIGN:")
    print("Technical passband (in Hz) : ", getFunctionalPassband(practicalBPF, cutoff=0.707))
    print("Technical passband ripple (V/V gain): ", getFunctionalRipple(practicalBPF, cutoff=0.707))
    print("Distortionless passband (in Hz) : ", getFunctionalPassband(practicalBPF, cutoff=0.99))
    print("Distortionless passband ripple (V/V gain): ", getFunctionalRipple(practicalBPF, cutoff=0.99))

    # ========================================================
    # CALCULATE STATS FOR THE PRACTICAL BPF (ANALOG DEVICES DESIGN)
    # ========================================================
    print("ANALOG DEVICES DESIGN:")
    print("Technical passband (in Hz) : ", getFunctionalPassband(analogDevicesBPF, cutoff=0.707))
    print("Technical passband ripple (V/V gain): ", getFunctionalRipple(analogDevicesBPF, cutoff=0.707))
    print("Distortionless passband (in Hz) : ", getFunctionalPassband(analogDevicesBPF, cutoff=0.99))
    print("Distortionless passband ripple (V/V gain): ", getFunctionalRipple(analogDevicesBPF, cutoff=0.99))

    # ========================================================
    # PLOT FREQUENCY RESPONSE FOR ALL FILTERS
    # ========================================================
    f_fft, proposed_value_filter_Hs = generateFilterResponse_v2(practicalBPF)
    f_fft, analog_devices_filter_Hs = generateFilterResponse_v2(analogDevicesBPF)
    plt.plot(f_fft, proposed_value_filter_Hs)
    plt.plot(f_fft, analog_devices_filter_Hs)
    plt.xscale('log')
    plt.show()

    # ========================================================
    # SIMULATE THE TIME DOMAIN OUTPUT WAVEFORM FOR ALL FILTERS
    # ========================================================
    # def simulateWfn(testWave, filterToApply, duration, plot=True)
    freqs, timeDomainOut = simulateWfn(practicalBPF, squareTest, simulationTime)
    freqs, timeDomainOutAnalog = simulateWfn(analogDevicesBPF, squareTest, simulationTime)

    plt.plot(np.ndarray.tolist(normalized(timeDomainOut))[0])
    plt.plot(np.ndarray.tolist(normalized(timeDomainOutAnalog))[0])
    plt.plot(np.ndarray.tolist(normalized(squareTest))[0])
    plt.show()

    # ========================================================
    # CALCULATE FFT COMPARISON PLOTS (PROPOSED VALUE DESIGN)
    # ========================================================
    ft_post_filter = np.abs(fft(timeDomainOut))
    total_length = len(ft_post_filter)
    ft_post_filter = np.ndarray.tolist(normalized(ft_post_filter, axis=0, order=2))[0][1:total_length - 1]
    ft_input_signal = np.abs(fft(squareTest))
    ft_input_signal = np.ndarray.tolist(normalized(ft_input_signal, axis=0, order=2))[0][1:total_length - 1]

    filter_freqs, ft_filter = np.abs(generateFilterResponse(practicalBPF, squareTest, simulationTime))
    ft_filter = np.ndarray.tolist(normalized(ft_filter, axis=0, order=2))[0][1:total_length - 1]

    freqs = freqs[1:total_length - 1]
    filter_freqs = filter_freqs[1:total_length - 1]

    plt.plot(freqs, ft_input_signal)
    plt.plot(freqs, ft_post_filter)
    plt.plot(filter_freqs, ft_filter)
    plt.xlim([500, 700000])
    plt.show()

    # ========================================================
    # CALCULATE FFT COMPARISON PLOTS (ANALOG DEVICES DESIGN)
    # ========================================================
    ft_post_filter_analog = np.abs(fft(timeDomainOutAnalog))
    total_length = len(ft_post_filter_analog)
    ft_post_filter_analog = np.ndarray.tolist(normalized(ft_post_filter_analog, axis=0, order=2))[0][1:total_length - 1]

    filter_freqs, ft_filter_analog = np.abs(generateFilterResponse(analogDevicesBPF, squareTest, simulationTime))
    ft_filter_analog = np.ndarray.tolist(normalized(ft_filter_analog, axis=0, order=2))[0][1:total_length - 1]

    filter_freqs = filter_freqs[1:total_length - 1]

    plt.plot(freqs, ft_input_signal)
    plt.plot(freqs, ft_post_filter_analog)
    plt.plot(filter_freqs, ft_filter_analog)
    plt.xlim([500, 700000])
    plt.show()

    # ========================================================
    # CALCULATE HAR PRE AND POST FILTERING (PROPOSED VALUE DESIGN)
    # ========================================================
    post_filter_peaks = signal.find_peaks(ft_post_filter, height=0.05)
    pre_filter_peaks = signal.find_peaks(ft_input_signal, height=0.05)
    pre_filter_fundamental = ft_input_signal[pre_filter_peaks[0][0]]
    pre_filter_harm = ft_input_signal[pre_filter_peaks[0][1]]
    post_filter_fundamental = ft_post_filter[post_filter_peaks[0][0]]
    post_filter_harm = ft_post_filter[post_filter_peaks[0][1]]

    post_filter_HAR = post_filter_fundamental / post_filter_harm
    pre_filter_HAR = pre_filter_fundamental / pre_filter_harm
    print("Pre-filter HAR (simulated): ", pre_filter_HAR)
    print("Post-filter HAR (simulated, proposed value design):", post_filter_HAR)

    # ========================================================
    # CALCULATE HAR PRE AND POST FILTERING (ANALOG DEVICES DESIGN)
    # ========================================================
    post_filter_peaks_analog = signal.find_peaks(ft_post_filter_analog, height=0.05)
    post_filter_fundamental_analog = ft_post_filter_analog[post_filter_peaks[0][0]]
    post_filter_harm_analog = ft_post_filter_analog[post_filter_peaks_analog[0][1]]

    post_filter_HAR_analog = post_filter_fundamental_analog / post_filter_harm_analog
    print("Post-filter HAR (simulated, Analog Devices design):", post_filter_HAR_analog)
