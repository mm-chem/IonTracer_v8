import numpy as np
import pickle
import pandas as pd
import statistics as stats
import traceback
import matplotlib.pyplot as plt
import csv
import math
import pickle
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
import scipy.signal as sig
import os
import itertools
import tkinter as tk
from tkinter import filedialog as fd
import v6_STFT_analysis as STFT
import v6_IonClass as Harmonics
from scipy import signal as sig
from scipy import stats as stats
from scipy import optimize as opt
import v1_TraceVisualizer as ZxxToolkit
import v1_2D_mass_spectrum_plotter as MSPlotter_2D
import v1_charge_loss_plotter as DropPlotter
import v1_drops_per_trace_plotter as DropsPerTrace
import v1_trace_slope_plotter as TraceSlopeDist
import v1_eV_per_charge_plotter as EnergyPlotter
import v1_trace_slope_plotter_no_drops as TraceSlopeDistNoDrops
import v1_ion_frequencies_plotter as IonFrequencies
import v1_mass_vs_freq_plotter as MassFreq
import v2_trap_dynamics_simulator as TrapDynamics
import v1_dynamics_charge_loss_plotter as Dynamics


def plot_rayleigh_line(axis_range=[0, 200]):
    coul_e = 1.6022E-19
    avo = 6.022E23
    surfacet = 0.07286
    # 0.07286 H20 at 20 C #0.05286 for NaCl split, 0.07286 for LiCl split 0.0179 for hexanes 0.0285 TOLUENE 0.050 for
    # mNBA
    surfacetalt = 0.0179
    perm = 8.8542E-12  # vacuum permittivity
    density = 999800  # g/m^3 999800 for water
    low_d, high_d, step_d = axis_range[0], axis_range[1], 0.5  # diameter range/step size in nm
    low = low_d * 1.0E-9
    high = high_d * 1.0E-9
    step = step_d * 1.0E-9
    qlist = []
    q2list = []
    mlist = []
    m_list = []
    for d in np.arange(low, high, step):
        q = (8 * math.pi * perm ** 0.5 * surfacet ** 0.5 * (d / 2) ** 1.5) / coul_e
        q2 = (8 * math.pi * perm ** 0.5 * surfacetalt ** 0.5 * (d / 2) ** 1.5) / coul_e
        qlist.append(q)
        q2list.append(q2)
        m = ((4 / 3) * math.pi * (d / 2) ** 3) * density * avo
        mlist.append(m)
        m_list.append(m)

    plt.plot(m_list, qlist, color='black', linestyle="dashed", linewidth=2)


def d_to_mass(diameter, density=1):
    mass = ((4 / 3) * np.pi * (0.5 * diameter) ** 3) * density
    return mass


def mass_to_d(mass, density=0.9988):
    # Assumes density is given in g/ml
    mass_g = mass / 6.022E23
    diameter_cm = (np.cbrt((mass_g / density) * (3 / 4) * (1 / np.pi))) * 2
    diameter_nm = diameter_cm * 10.0E6
    return diameter_nm


def returnRayleighLine(min_mass, max_mass, number_of_points):
    coul_e = 1.6022E-19
    avo = 6.022E23
    surfacet = 0.07286
    # 0.07286 H20 at 20 C #0.05286 for NaCl split, 0.07286 for LiCl split 0.0179 for hexanes 0.0285 TOLUENE 0.050 for
    # mNBA
    surfacetalt = 0.0179
    perm = 8.8542E-12  # vacuum permittivity
    density = 999800  # g/m^3 999800 for water
    low_d, high_d = mass_to_d(min_mass), mass_to_d(max_mass)  # diameter range/step size in nm
    step_d = (high_d - low_d) / number_of_points
    low = low_d * 1.0E-9
    high = high_d * 1.0E-9
    step = step_d * 1.0E-9
    qlist = []
    q2list = []
    mlist = []
    m_list = []
    for d in np.arange(low, high, step):
        q = (8 * math.pi * perm ** 0.5 * surfacet ** 0.5 * (d / 2) ** 1.5) / coul_e
        q2 = (8 * math.pi * perm ** 0.5 * surfacetalt ** 0.5 * (d / 2) ** 1.5) / coul_e
        qlist.append(q)
        q2list.append(q2)
        m = ((4 / 3) * math.pi * (d / 2) ** 3) * density * avo
        mlist.append(m)
        m_list.append(m)
    return m_list, qlist


def sliding_window_diff(trace, width, step=1):
    winStartIndex = 0
    winEndIndex = 0 + width
    slidingDiff = np.zeros([int(np.floor(len(trace) / step)) - 1])
    for i in range(len(slidingDiff) - 1):
        avgBefore = np.average(trace[winStartIndex:winEndIndex])
        winStartIndex = winStartIndex + step
        winEndIndex = winEndIndex + step
        avgAfter = np.average(trace[winStartIndex:winEndIndex])
        slidingDiff[i] = avgAfter - avgBefore
    return slidingDiff


class Trace:
    def __init__(self, filePath, spamm, drop_threshold, UTID, lock_init_freq=False, fixed_eV=0):
        self.lock_init_freq = lock_init_freq
        self.fixed_eV = fixed_eV
        self.SPAMM = spamm
        self.avg_slope = 0
        self.Zxx_background = []
        self.folder = filePath.rsplit('\\', 1)[0]
        self.file = filePath.rsplit('\\', 1)[-1]
        self.UTID = UTID
        self.higher_harmonics = []
        self.trace = []
        self.initial_frequency = 0
        self.trace_magnitudes = []
        self.trace_harm = []
        self.trace_magnitudes_harm = []
        self.fragmentation_indices = []
        self.fragments = []
        self.drops = []
        self.trace_indices = []
        self.drop_threshold = drop_threshold
        self.avg_mass = 0
        self.avg_charge = 0
        try:
            with open(filePath, newline='') as csvfile:
                traceReader = csv.reader(csvfile, delimiter=',')
                traceData = []
                for row in traceReader:
                    traceData.append(row)

            # Assumes file with format HEADER, //, FREQ, //, AMP, //, FREQ_HARM, //, AMP_HARM
            # Produces traceData list: [0] = HEADER, [1] = FREQ[i], [2] = AMP[i], [3] = HARM_AMP[i]
            traceData = [list(y) for x, y in itertools.groupby(traceData[0], lambda z: z == ' //') if not x]
            self.startFreq = float(traceData[0][0].split("|")[0])
            f_reso = float(traceData[0][0].split("|")[1])
            self.time_offset = float(traceData[0][0].split("|")[2])
        except Exception as e:
            print("Error opening file" + str(filePath) + ": ", e)

        try:
            self.trace = [float(i) * f_reso + self.startFreq for i in traceData[1]]
            self.trace_magnitudes = [float(i) for i in traceData[2]]
            self.trace_harm = [float(i) * 2 for i in self.trace]
            self.trace_magnitudes_harm = [float(i) for i in traceData[3]]
        except Exception:
            print("Error reading phase-included file. Retrace raw data with v7 tracer")

        try:
            self.strip_bad_endpoints()
            self.trace_indices = np.array(range(len(self.trace)))
            self.initial_frequency = self.trace[0]
            self.fragment_trace()
            for frag in self.fragments:
                self.avg_mass = self.avg_mass + frag.mass
            self.avg_mass = self.avg_mass / len(self.fragments)
            for frag in self.fragments:
                self.avg_charge = self.avg_charge + frag.charge
            self.avg_charge = self.avg_charge / len(self.fragments)
        except Exception as e:
            print("Error dealing with fragmented traces: ", e)
            traceback.print_exc()

        try:
            self.calculate_avg_slope()
        except Exception as e:
            print('Error encountered finding avg slope: ', e)

    def calculate_avg_slope(self):
        # In Hz/s drift
        slope_sum = 0
        counter = 0
        for frag in self.fragments:
            slope_sum += frag.linfitEquation.coefficients[0]
            counter += 1
        self.avg_slope = slope_sum / counter

    def strip_bad_endpoints(self, range_HAR=[1.7, 2.7]):
        is_bad_point = True
        index = -1
        initial_trace_len = len(self.trace)
        while is_bad_point:
            test_HAR = self.trace_magnitudes[index] / self.trace_magnitudes_harm[index]
            if test_HAR <= range_HAR[0] or test_HAR >= range_HAR[1]:
                # Pop off point from trace
                is_bad_point = True
                self.trace.pop(index)
                self.trace_magnitudes.pop(index)
                self.trace_magnitudes_harm.pop(index)
                self.trace_harm.pop(index)
                index = index - 1
            else:
                is_bad_point = False
                # print("Stripped endpoints: " + str(abs(index) - 1) + " (" + str(int(((abs(index) - 1)/initial_trace_len) * 100)) + " %)")

    def bridge_fragments(self, frag1, frag2, drop_counter):
        gap_length = frag2.fragStart - frag1.fragEnd
        new_drop_point = frag1.fragEnd + np.floor(gap_length / 2)
        frag1.fragEnd = new_drop_point
        frag2.fragStart = new_drop_point
        initial_C_E = frag1.C_E
        final_C_E = frag2.C_E
        try:
            startFreq = frag1.linfitEquation(new_drop_point)
            endFreq = frag2.linfitEquation(new_drop_point)
            harmInitFreq = frag1.harmLinfitEquation(new_drop_point)
            harmFinalFreq = frag2.harmLinfitEquation(new_drop_point)
            startMag = frag1.avg_mag
            endMag = frag2.avg_mag
            startCharge = frag1.charge
            endCharge = frag2.charge
            C_E_initial = initial_C_E
            C_E_final = final_C_E
            t_before = len(frag1.trace)
            t_after = len(frag2.trace)
            start_mass = frag1.mass
            end_mass = frag2.mass
            fundamental_trace = self.trace
            drop_index = frag1.fragEnd
            trace_indices = self.trace_indices
            folder = self.folder
            UDID = str(self.UTID) + "d" + str(drop_counter)
            newDrop = Drop(startFreq, endFreq, startMag, endMag, startCharge, endCharge, C_E_initial, C_E_final,
                           t_before, t_after, start_mass, end_mass, fundamental_trace, drop_index,
                           trace_indices, folder, harmInitFreq, harmFinalFreq, self.UTID, UDID, frag1.energy_eV)
            self.drops.append(newDrop)
        except Exception as e:
            print("Error bridging fragments...", e)
            traceback.print_exc()

    def plot_ion_traces(self, fragLines=None, harm=None):
        if fragLines is None:
            fragLines = []
        plt.plot(self.trace_indices, self.trace)
        for frag in self.fragments:
            plt.plot(frag.trace_indices, frag.trace)
        if harm:
            plt.plot(self.trace_indices, self.trace_harm)
        for line in fragLines:
            plt.axvline(x=line + self.time_offset, color="green")
        plt.show()

    def drop_threshold_scaler(self, freq):
        # This is a quadratic function to dynamically change the drop threshold depending on frequency
        f_scaled_noise_limit = -((freq - self.startFreq) / self.startFreq) ** 2 + self.drop_threshold
        f_scaled_noise_limit = self.drop_threshold
        return f_scaled_noise_limit

    def fragment_trace(self, trigger_source='harm'):
        f_scale_factor = self.trace[0]
        self.drop_threshold = self.drop_threshold_scaler(f_scale_factor)
        front_truncation = 3
        if trigger_source == 'harm':
            differentialTrace = sliding_window_diff(self.trace_harm, 3)
        else:
            differentialTrace = sliding_window_diff(self.trace, 3)
        self.fragmentation_indices.append(0)
        # print('Drop threshold: ', str(self.drop_threshold))

        plot = 0
        min_drop_spacing = 5
        intercept = abs(stats.siegelslopes(differentialTrace)[1])
        differentialTrace = differentialTrace - intercept
        peak_indexes, check_prop = sig.find_peaks(-differentialTrace, height=-self.drop_threshold,
                                                  distance=min_drop_spacing)
        for index in peak_indexes:
            self.fragmentation_indices.append(index)
            # plot = 1

        self.fragmentation_indices.append(len(self.trace) - 1)
        # Fragment builder splits the trace up into fragments and creates fragment objects
        self.fragment_builder(front_truncation)
        # Fragment analyzer cuts out bad fragments and stitches the good ones together
        self.fragment_analyzer()
        if plot == 1:
            print(self.fragmentation_indices)
            self.plot_ion_traces(self.fragmentation_indices)
            plt.show()
            # plt.plot(-differentialTrace)
            # plt.show()

    def fragment_builder(self, front_truncation):
        fragment_counter = 0
        for i in range(len(self.fragmentation_indices) - 1):
            fragStart = self.fragmentation_indices[i] + int(front_truncation)
            fragEnd = self.fragmentation_indices[i + 1]
            # If the fragment has more than 2 points in it (it should.....)
            if fragEnd - fragStart > 2:
                fragTrace = self.trace[fragStart:fragEnd]
                fragIndices = self.trace_indices[fragStart:fragEnd]
                harmFragTrace = self.trace_harm[fragStart:fragEnd]
                fragMag = self.trace_magnitudes[fragStart:fragEnd]
                harmFragMag = self.trace_magnitudes_harm[fragStart:fragEnd]
                UFID = int(str(self.UTID) + str(fragment_counter))
                newFrag = Fragment(fragTrace, harmFragTrace, fragMag, harmFragMag, fragStart, fragEnd,
                                   self.SPAMM, fragIndices, self.UTID, UFID, self.lock_init_freq, self.fixed_eV)
                fragment_counter = fragment_counter + 1
                self.fragments.append(newFrag)
            else:
                print("Rejected single point ion fragment.")

    def fragment_analyzer(self):
        if len(self.fragments) > 1:
            useful_fragments = []
            drop_counter = 0
            for i in range(len(self.fragments) - 1):
                delta_x = self.fragments[i + 1].fragStart - self.fragments[i].fragEnd
                # # Should be in BINS (ideally)
                if delta_x <= 5:
                    self.bridge_fragments(self.fragments[i], self.fragments[i + 1], drop_counter)
                    drop_counter = drop_counter + 1
                    useful_fragments.append(self.fragments[i])
                    if i == len(self.fragments) - 2:
                        # Add the last fragment too (if we are at the end of the line)
                        useful_fragments.append(self.fragments[i + 1])
            self.fragments = useful_fragments


class Fragment:
    def __init__(self, fragTrace, harmFragTrace, fragMag, harmFragMag, fragStart, fragEnd, spamm, trace_indices,
                 UTID, UFID, lock_init_freq, fixed_eV):
        self.lock_init_freq = lock_init_freq
        self.fixed_eV = fixed_eV
        self.UTID = UTID
        self.UFID = UFID
        self.charge_pt_by_pt = []
        self.mass_pt_by_pt = []
        self.HAR_pt_by_pt = []
        self.C_E_pt_by_pt = []
        self.energy_eV_pt_by_pt = []
        self.TTR_from_HAR_pt_by_pt = []
        self.pt_by_pt_mass_slope = None
        self.pt_by_pt_mass_intercept = None
        self.m_z_pt_by_pt = []
        self.mass_variance_corrected = None
        self.SPAMM = spamm
        self.trace_indices = trace_indices  # Fragment indices
        self.fragStart = fragStart
        self.fragEnd = fragEnd
        self.trace = fragTrace
        self.energy_eV = None
        self.harm_trace = harmFragTrace
        self.frag_mag = fragMag
        self.avg_mag = np.average(fragMag)
        self.harm_frag_mag = harmFragMag
        self.harm_avg_mag = np.average(harmFragMag)
        self.avg_freq = np.average(fragTrace)
        self.HAR = self.avg_mag / self.harm_avg_mag
        for n in range(len(self.frag_mag)):
            tentative_HAR = self.frag_mag[n] / self.harm_frag_mag[n]
            self.HAR_pt_by_pt.append(tentative_HAR)
        self.C_E = None
        self.m_z = None
        self.mass = None
        self.charge = None
        self.linfitEquation = None
        self.harmLinfitEquation = None
        self.x_axis = np.linspace(self.fragStart, self.fragEnd, len(self.trace))

        self.lin_fit_frag()
        self.lin_fit_frag_harm()
        self.magic(self.lock_init_freq, self.fixed_eV)

        # Compute uncertainty in charge measurement for the fragment
        self.charge_std = np.std(self.charge_pt_by_pt)
        # 25 IS THE SEGMENT LENGTH...... ONLY VALID IF THIS IS ACCURATE
        self.charge_uncertainty = self.charge_std / np.sqrt(len(self.trace) / 5)

        # Compute uncertainty in charge measurement for the fragment
        try:
            ev_fit = np.polyfit(range(len(self.trace)), self.energy_eV_pt_by_pt, 1)
            self.ev_linfit_equation = np.poly1d(ev_fit)
            slope = self.ev_linfit_equation.coefficients[0]
            intercept = self.ev_linfit_equation.coefficients[1]

            self.pt_by_pt_ev_slope = slope
            self.pt_by_pt_ev_intercept = intercept
            self.delta_ev = self.pt_by_pt_ev_intercept - (
                    self.pt_by_pt_ev_slope * len(self.trace) + self.pt_by_pt_ev_intercept)
        except Exception:
            print('Energy fit error')
        # This doesnt seem to work... maybe because you cant calculate variance on something changing??
        # Attempting to calculate the residuals from the fit equation first, then calculate the variance of THAT
        try:
            slope_corrected_mass_pt_by_pt = []
            # Calculate this fit with respect to indices started from wherever this fragment starts
            # Note that this will start trace elements at 0 regardless of their actual time of existence
            mass_fit = np.polyfit(range(len(self.trace)), self.mass_pt_by_pt, 1)
            self.mass_linfit_equation = np.poly1d(mass_fit)
            slope = self.mass_linfit_equation.coefficients[0]
            intercept = self.mass_linfit_equation.coefficients[1]

            self.pt_by_pt_mass_slope = slope
            self.pt_by_pt_mass_intercept = intercept

            for n in range(len(self.trace)):
                slope_corrected_mass_pt_by_pt.append(self.mass_pt_by_pt[n] - slope * n)
            self.mass_variance_corrected = np.std(slope_corrected_mass_pt_by_pt)
            self.mass_uncertainty_corrected = self.mass_variance_corrected / np.sqrt(len(self.trace) / 25)

            # arr = np.matrix(self.mass_pt_by_pt)
            # name_string = '/Users/mmcpartlan/Desktop/' + str(self.charge) + 'Da_pt_by_pt.csv'
            # pd.DataFrame(arr).to_csv(name_string)

            # print(self.mass, self.pt_by_pt_mass_slope)
            # plt.plot(self.charge_pt_by_pt)
            # plt.plot(slope_corrected_charge_pt_by_pt)
            # plt.plot(self.mass_linfit_equation(range(len(self.trace))))
            # plt.show()
            # plt.plot(self.energy_eV_pt_by_pt)
            # plt.plot(self.ev_linfit_equation(range(len(self.trace))))
            # plt.show()
        except Exception as e:
            print("Error calculating corrected mass variance...", e)

        self.mass_variance = np.std(self.mass_pt_by_pt)
        # 25 IS THE SEGMENT LENGTH...... ONLY VALID IF THIS IS ACCURATE
        self.mass_uncertainty = self.mass_variance / np.sqrt(len(self.trace) / 25)

    def lin_fit_frag(self):
        if len(self.trace) > 1:
            fit = np.polyfit(self.x_axis, self.trace, 1)
            self.linfitEquation = np.poly1d(fit)
        else:
            print("Unable to fit line of " + str(len(self.trace)))

    def lin_fit_frag_harm(self):
        if len(self.harm_trace) > 1:
            fit = np.polyfit(self.x_axis, self.harm_trace, 1)
            self.harmLinfitEquation = np.poly1d(fit)
        else:
            print("Unable to fit line of " + str(len(self.harm_trace)))

    def magic(self, lock_init_freq=False, fixed_eV=0):
        if self.SPAMM == 2:
            trap_V = 336.6  # trapping cones potential

            # Trap Potential/Ion eV/z ratio to energy calibration (all trap potentials)
            # Equation---Energy =Trap/(A*TTR^3 + B*TTR^2 + C*TTR + D)
            A = -0.24516
            B = 1.84976
            C = -4.92709
            D = 5.87484

            # HAR to TTR calibration (all trap settings)
            # Equation--- TTR = E*HAR^3 + F*HAR^2 + G*HAR + H
            E = -0.5305
            F = 4.0047
            G = -10.535
            H = 11.333

            # Raw Amplitude to Charge Calibration (via BSA calibration curve, may change/improve with more calibration data)
            # Equation--- Charge = (Raw Amplitude + J)/K
            J = 0.0000
            # K = 0.91059  # OLD K VAL
            K = 1.126  # K-VAL calibrated from vaults on Nov 19, 2024 (with new filter)
            # K = 0.81799  # K VAL from 4/22 (calibrated by Zach)
            # K = 0.5500     # NOT a valid calibration value, just used for testing
            # K = 0.9999

            # (amp per charge) 1.6191E-6 for 250 ms zerofill, 1.6126E-6 for 125 ms zerofill
            # calibration value with filter added 1.3342E-6 (0.91059 with updated amplitude in analysis program)
            # calibration value with butterworth bandpass + old filter in series 1.4652E-6
            # calibration value * rough factor in analysis program (682500 currently) 0.999999
            # current calibration value ~0.87999 with 682500 factor in analysis
            # 12-20 calibration value 0.84799
            # 2-14 calibration value 0.81799

            # Charge correction factor constants from simulation of 20 kHz 400 us RC simulation across range of TTR
            # Equation--- Factors = 1/(L*TTR+M)
            L = -0.1876
            M = 1.3492

            # Energy (eV/z) to C-value conversion (m/z = C(E)/f^2)
            # C(E) function of both trap_V and energy (but not their ratio)
            # Equation--- C(E) = (A00+A10*Energy+A01*Trap_V+A20*Energy^2+A11*Energy*Trap_V+A02*Trap_V^2+A30*Energy^3+
            # A21*Energy^2*Trap_V+  A12*Energy*Trap_V^2+A03*Trap_V^3)^2
            A00 = 602227.450695831
            A10 = -11874.0933576314
            A01 = 15785.4833447021
            A20 = -110.631481581344
            A11 = 179.897639821121
            A02 = -80.623006669759
            A30 = -0.0729582264231568
            A21 = 0.3250825276845
            A12 = -0.369837273160484
            A03 = 0.130371575432137

        if self.SPAMM == 3:
            A = 75.23938332
            B = -11.77882363
            C = 2.75911194
            D = 0.85939469
            E = 0.819414
            F = -3.62893
            G = 5.541968
            H = -2.67478
            J = 0.0000
            K = 0.8272
            L = 4.78043467
            M = 0.13541854
            A00 = 783474.415
            A10 = -18962.5956
            A01 = 21519.0704
            A20 = -117.974195
            A11 = 203.573934
            A02 = -93.0612998
            A30 = -0.0857022781
            A21 = 0.365994533
            A12 = -0.418614264
            A03 = 0.147667014
            B00 = -0.38278304
            B10 = 1.13696746
            B01 = -1.91505933
            B20 = -0.00125747556
            B11 = 0.01133092
            B02 = -0.0249704296
            B30 = 0.00000298600818
            B21 = -0.0000331707251
            B12 = 0.00011459605
            B03 = -0.000119888859
            S = 0.024725
            T = -0.232900
            U = 0.965265
            trap_V = 330

        for n in range(len(self.frag_mag)):
            self.TTR_from_HAR_pt_by_pt.append(E * self.HAR_pt_by_pt[n] ** 3 + F * self.HAR_pt_by_pt[n] ** 2 + \
                                              G * self.HAR_pt_by_pt[n] + H)

            if fixed_eV == 0:
                self.energy_eV_pt_by_pt.append(trap_V / (A * self.TTR_from_HAR_pt_by_pt[n] ** 3 + B *
                                                         self.TTR_from_HAR_pt_by_pt[n] ** 2 + C *
                                                         self.TTR_from_HAR_pt_by_pt[n] + D))
            else:
                self.energy_eV_pt_by_pt.append(fixed_eV)

            self.C_E_pt_by_pt.append((A00 + A10 * self.energy_eV_pt_by_pt[n] + A01 * trap_V + A20 *
                                      self.energy_eV_pt_by_pt[n] ** 2 + A11 * self.energy_eV_pt_by_pt[n] *
                                      trap_V + A02 * trap_V ** 2 + A30 * self.energy_eV_pt_by_pt[n] ** 3 +
                                      A21 * self.energy_eV_pt_by_pt[n] ** 2 * trap_V + A12 *
                                      self.energy_eV_pt_by_pt[n] * trap_V ** 2 + A03 * trap_V ** 3) ** 2)

            if lock_init_freq:
                initial_freq = self.trace[0]
                for x in range(len(self.trace)):
                    self.trace[x] = initial_freq

            self.m_z_pt_by_pt.append(self.C_E_pt_by_pt[-1] / self.trace[n] ** 2)
            uncorrected_charge = (self.frag_mag[n] + J) / K  # calibration from BSA charge states
            corr_factors = 1 / (L * self.TTR_from_HAR_pt_by_pt[n] + M)
            self.charge_pt_by_pt.append(uncorrected_charge * corr_factors)
            self.mass_pt_by_pt.append(self.m_z_pt_by_pt[-1] * self.charge_pt_by_pt[-1])

        self.energy_eV = np.average(self.energy_eV_pt_by_pt)
        self.C_E = np.average(self.C_E_pt_by_pt)
        self.m_z = np.average(self.m_z_pt_by_pt)
        self.charge = np.average(self.charge_pt_by_pt)
        self.mass = np.average(self.mass_pt_by_pt)


class Drop:
    def __init__(self, startFreq, endFreq, startMag, endMag, startCharge, endCharge, initial_C_E, final_C_E, t_before,
                 t_after, start_mass, end_mass, fundamental_trace, drop_index, trace_indices, folder, harmInitFreq,
                 harmFinalFreq, UTID, UDID, energy_eV_z):
        # Used to filter out super short drops from analyis
        self.energy_eV_z = energy_eV_z
        self.UDID = UDID
        self.UTID = UTID
        self.fundamental_trace = fundamental_trace
        self.folder = folder
        self.trace_indices = trace_indices
        self.drop_index = drop_index
        self.drop_index_trace = trace_indices[int(drop_index)]
        self.t_before = t_before
        self.t_after = t_after
        self.startCharge = startCharge
        self.endCharge = endCharge
        self.delta_charge = endCharge - startCharge
        self.start_mass = start_mass
        self.end_mass = end_mass
        self.avg_mass = np.average([start_mass, end_mass])
        # charge_error_guess = np.average([startCharge, endCharge]) * 0.1  # Used to check if charge error can shift
        # the distribution
        self.avg_charge = np.average([startCharge, endCharge])  # + charge_error_guess
        self.freq_change_magnitude = endFreq - startFreq

        self.charge_change_magnitude = endMag - startMag  # Unused in current calculations.
        self.initialFreq = startFreq
        self.finalFreq = endFreq
        self.f_squared_ratio_change = (startFreq ** 2) / (endFreq ** 2)
        self.f_squared_ratio_change = (harmInitFreq ** 2) / (harmFinalFreq ** 2)
        self.freq_computed_charge_loss = -(self.f_squared_ratio_change - 1) * self.avg_charge

        # Lets figure out what m/z change this drop accounts for...
        # Check change in energy. If approximately constant, we can say delta_m/z = C_E / delta_f^2
        self.initial_C_E = initial_C_E
        self.final_C_E = final_C_E
        self.delta_C_E = final_C_E - initial_C_E
        self.delta_C_E_percent = (self.delta_C_E / self.initial_C_E) * 100
        # Calculate delta_m/z = C_E / delta_f^2
        self.delta_m_z = self.final_C_E / (self.finalFreq ** 2) - self.final_C_E / (
                self.initialFreq ** 2)  # Observed m/z change
        self.scaled_m_z = (self.delta_m_z / (self.avg_mass / self.avg_charge)) * 100
        self.expected_1C_mz_change = (self.avg_mass / (self.avg_charge - 1)) - (
                self.avg_mass / self.avg_charge)  # The expected m/z change (positive) due to single charge loss
        self.C_loss_scaled_m_z = self.delta_m_z / self.expected_1C_mz_change
        self.delta_mass = self.delta_m_z * self.avg_charge

        # Experimental charge loss calculation (TrapDynamics incorporation)
        # Tracing the minimum abs value freq change line...
        # EDIT DYNAMICS SETTINGS IN THE DYNAMICS PROGRAM
        trap_dynamics_ion = TrapDynamics.ExportInterface(self.start_mass, self.startCharge, self.energy_eV_z,
                                                         fixed_delta_mass=None)
        self.dynamics_computed_delta_z_min = trap_dynamics_ion.min_abs_freq_change_curve(self.freq_change_magnitude)
        self.dynamics_computed_delta_z_max = trap_dynamics_ion.max_abs_freq_change_curve(self.freq_change_magnitude)


def gauss(x, A, mu, sigma, offset):
    return offset + A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


if __name__ == "__main__":
    # Uncomment for bulk analysis (point to a folder, then pool all .trace files into one analysis
    ################################################################
    # trace_folders = STFT.choose_top_folders(".traces")
    # print(trace_folders)
    # file_ending = ".trace"
    # filelists = STFT.generate_filelist(trace_folders, file_ending)
    # analysis_name = trace_folders[0].split('.pool', maxsplit=1)[0]
    # new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]
    # analysis_name = analysis_name + '.figures'
    ################################################################

    # Uncomment for single folder analysis (select the .traces folder manually)
    ################################################################
    win = tk.Tk()
    win.focus_force()
    win.withdraw()
    folder = fd.askdirectory(title="Choose traces folder")
    print(folder)
    filelists = STFT.generate_filelist([folder], ".tv7f")
    file_count = len(filelists[0])
    analysis_name = folder.rsplit('.', maxsplit=1)[0]
    new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]
    noncrit_zero_div_errors = 0

    ################################################################

    save_plots = True
    # Define font params for exported plots
    SMALL_SIZE = 24
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 36

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    show_plots = False
    smoothed_output = True  # Smooth the histogram before calculating the peak
    f_computed_axv_lines = True
    SPAMM = 2
    print("ANALYSIS PERFORMED FOR SPAMM INSTRUMENT")
    print("---------------------------------------")
    print(str(SPAMM))
    print("---------------------------------------")
    drop_threshold = -15  # NOTE: This parameter is affected by the K parameter
    # PLOT SELECTION CONTROLS:
    dynamics_charge_loss = 1
    freq_vs_drop_events = 0
    ion_frequencies = 1
    drops_per_trace = 1
    mass_freq = 1
    trace_slope_distribution = 1
    freq_drop_magnitude = 1
    f_computed_charge_loss = 1
    amp_computed_charge_loss = 1
    delta_2D_mass_charge = 0
    C_E_percent_change = 0
    m_z_drop_1D_spectrum = 0
    HAR_eV_distribution = 1
    mass_spectrum_2D = 1
    plot_drop_statistics = 1
    plot_1C_loss_scaled_m_z = 0
    # For Emeline's project
    export_no_drop_slopes = 1
    export_drop_files = 1
    z_2_n = 0

    # Salt and z2/n controls
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    salt_n = 2
    # salt_MW = 245.2  # LaCl3
    # salt_MW = 74.5  # KCl
    # salt_MW = 111.1  # CaCl2
    salt_MW = 58.5  # NaCl

    # Energy filter controls
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # eV / z boundaries... ions cannot physically exist outside a small range of energies. Set that range here
    ev_z_min = 175  # Default 200
    ev_z_max = 245  # Default 245

    # Splitting data by slope controls
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    separation_line_slopes = [-100, 100]  # Dividing line between slopes... set to zero for no slope separation
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Ion existence pre/post emission event controls
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    before_existence_threshold = 15
    after_existence_threshold = 15
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Mass filter controls
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    max_mass = 40 * 1000000  # Maximum mass in MDa (only adjust 1st number)
    min_mass = 0 * 1000000  # Minimum mass in MDa (only adjust 1st number)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Charge filter controls
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    max_charge_selection = 1000
    min_charge_selection = 5
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # HAR Allowed: 1.65 - 2.55

    # Drop size filter controls
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # If we only want to look at traces that contain a drop in a specific size range (in freq-computed range),
    # define that range here. Otherwise, set to -20 and +20 (UNITS ARE CHARGES)
    min_drop_search_boundary = -200
    max_drop_search_boundary = 200

    known_noise_peaks = [18000, 67500]
    deviation_max_bipolar = 300

    analysis_name = analysis_name + "_" + str(int(min_mass / 1000000)) + "_" + str(int(max_mass / 1000000)) + "MDa"
    # analysis_name = (analysis_name + "_" + str(min_drop_search_boundary) + "_" + str(max_drop_search_boundary) +
    #                  "_drops_" + str(before_existence_threshold) + "_spacing_" + str(drop_threshold) + "_threshold")
    fig_save_dir = analysis_name + '.tv7fg'
    analysis_name = analysis_name + '.tv7p'

    try:
        os.mkdir(analysis_name)
    except FileExistsError:
        print("Path exists already.")
    analysis_name = analysis_name + '/' + new_folder_name

    fail_count_slope = 0
    fail_count_mass = 0
    fail_count_charge = 0
    fail_count_drop = 0
    fail_count_energy = 0
    fail_count_noise = 0

    traces = []
    initial_legit_traces = len(traces)
    drops = []
    file_counter = 0
    for file in filelists[0]:
        print("Processing file " + str(file_counter) + " of " + str(file_count))
        newTrace = Trace(file, SPAMM, drop_threshold, file_counter, lock_init_freq=False, fixed_eV=0)
        file_counter = file_counter + 1
        traces.append(newTrace)
        if len(newTrace.drops) > 0:
            for element in newTrace.drops:
                if element.t_before > before_existence_threshold:
                    if element.t_after > after_existence_threshold:
                        drops.append(element)

    drop_counts = []
    filtered_traces = []
    filtered_drops = []

    mass_collection = []
    charge_collection = []
    m_z_collection = []
    energy_collection = []
    HAR_collection = []
    included_slopes = []
    no_drop_traces = []
    trace_lifetime = []
    fragment_lifetime = []
    initial_freq_collection = []

    for trace in traces:
        is_included = True
        fragment_counter = 0
        avg_charge_frags = 0
        avg_mass_frags = 0
        avg_mz_frags = 0
        avg_energy_frags = 0
        avg_HAR_frags = 0
        avg_slope_frags = 0
        avg_frag_length = 0
        trace_length = len(trace.trace)

        for fragment in trace.fragments:
            # Average the mass, charge, HAR, energy, and m/z of all fragments in a trace
            fragment_counter = fragment_counter + 1
            avg_charge_frags = avg_charge_frags + fragment.charge
            avg_mass_frags = avg_mass_frags + fragment.mass
            avg_mz_frags = avg_mz_frags + fragment.m_z
            avg_energy_frags = avg_energy_frags + fragment.energy_eV
            avg_HAR_frags = avg_HAR_frags + fragment.HAR
            avg_frag_length = avg_frag_length + len(fragment.trace)
            try:
                avg_slope_frags = avg_slope_frags + (
                        fragment.linfitEquation.coefficients[0] ** 2 / fragment.linfitEquation.coefficients[1] ** 2)
            except Exception as e:
                print('Linear fit produced no slope')

        try:
            avg_charge_frags = avg_charge_frags / fragment_counter
            avg_mass_frags = avg_mass_frags / fragment_counter
            avg_mz_frags = avg_mz_frags / fragment_counter
            avg_energy_frags = avg_energy_frags / fragment_counter
            avg_HAR_frags = avg_HAR_frags / fragment_counter
            avg_slope_frags = avg_slope_frags / fragment_counter
            avg_slope_frags = avg_slope_frags * 10.0e7
            avg_frag_length = avg_frag_length / fragment_counter
        except ZeroDivisionError:
            print("ERROR (noncritical): NO fragments in trace / division by zero")
            noncrit_zero_div_errors = noncrit_zero_div_errors + 1

        # If the trace has a slope that is out of bounds, don't include it
        if separation_line_slopes[0] > avg_slope_frags or avg_slope_frags > separation_line_slopes[1]:
            is_included = False
            print('Rejected trace: Slope out of bounds (' + str(avg_slope_frags) + ')')
            fail_count_slope = fail_count_slope + 1
        # If the trace has an avg mass that is out of bounds, don't include it either.
        if min_mass > trace.avg_mass or trace.avg_mass > max_mass:
            is_included = False
            print('Rejected trace: Mass out of bounds (' + str(trace.avg_mass) + ')')
            fail_count_mass = fail_count_mass + 1
        # If the trace has an avg charge that is out of bounds, don't include it either.
        if min_charge_selection > trace.avg_charge or trace.avg_charge > max_charge_selection:
            is_included = False
            print('Rejected trace: Charge out of bounds (' + str(trace.avg_charge) + ')')
            fail_count_charge = fail_count_charge + 1
        # If the energy ev/z is out of range, do not include it
        if ev_z_min > avg_energy_frags or ev_z_max < avg_energy_frags:
            is_included = False
            print('Rejected trace: Energy out of bounds (' + str(avg_energy_frags) + ')')
            fail_count_energy = fail_count_energy + 1
        # If the trace occurs at a known noise peak, dont include it
        for peak in known_noise_peaks:
            if abs(np.mean(trace.trace) - peak) <= deviation_max_bipolar:
                is_included = False
                print('Rejected trace: Known noise peak (' + str(np.mean(trace.trace)) + ')')
                fail_count_noise = fail_count_noise + 1

        if is_included:
            if len(trace.fragments) == 1:
                no_drop_traces.append(trace)
            filtered_traces.append(trace)
            mass_collection.append(avg_mass_frags)
            charge_collection.append(avg_charge_frags)
            m_z_collection.append(avg_mz_frags)
            energy_collection.append(avg_energy_frags)
            HAR_collection.append(avg_HAR_frags)
            included_slopes.append(avg_slope_frags)
            trace_lifetime.append(trace_length)
            fragment_lifetime.append(avg_frag_length)
            initial_freq_collection.append(trace.initial_frequency)
            if len(trace.drops) > 0:
                added_drop = False
                for element in trace.drops:
                    if element.t_before > before_existence_threshold:
                        if element.t_after > after_existence_threshold:
                            if min_drop_search_boundary < element.freq_computed_charge_loss < max_drop_search_boundary:
                                filtered_drops.append(element)
                                added_drop = True
                if added_drop:
                    drop_counts.append(len(trace.drops))
                else:
                    drop_counts.append(0)
            else:
                drop_counts.append(0)

    fail_count_drop = len(drops) - len(filtered_drops)
    traces = filtered_traces
    drops = filtered_drops

    no_drop_initial_mass = []
    no_drop_final_mass = []
    no_drop_slopes = []
    delta_mass_no_drops = []
    no_drop_avg_lifetime = []
    for sloped_trace in no_drop_traces:
        no_drop_final_mass.append(sloped_trace.fragments[0].pt_by_pt_mass_slope * len(sloped_trace.fragments[0].trace)
                                  + sloped_trace.fragments[0].pt_by_pt_mass_intercept)
        no_drop_initial_mass.append(sloped_trace.fragments[0].pt_by_pt_mass_intercept)
        delta_mass_no_drops.append(no_drop_initial_mass[-1] - no_drop_final_mass[-1])
        no_drop_avg_lifetime.append(len(sloped_trace.trace))
        try:
            no_drop_slopes.append((sloped_trace.fragments[0].linfitEquation.coefficients[0] ** 2 /
                                   sloped_trace.fragments[0].linfitEquation.coefficients[1] ** 2) * 10.0e7)
        except Exception as e:
            print('Linear fit produced no slope')

    a1_init_mass = np.array(no_drop_initial_mass)
    a2_delta_mass = np.array(delta_mass_no_drops)

    if export_no_drop_slopes:
        dbfile = open(str(analysis_name) + '_no_drop_slopes.pickle', 'wb')
        pickle.dump(no_drop_slopes, dbfile)
        dbfile.close()
        if save_plots:
            TraceSlopeDistNoDrops.plotter(fig_save_dir)

    dropsSquaredRatioChange = []
    dropsMagnitude = []
    dropsCharge = []
    dropsChargeChange = []  # Amplitude computed
    freqComputedChargeLoss = []  # Frequency computed
    delta_m_z = []
    C_loss_scaled_m_z = []
    scaled_delta_m_z = []
    delta_C_E = []
    delta_C_E_percent = []
    delta_mass = []
    delta_charge = []  # Frequency computed here, amplitude computed in drop objects
    dynamics_computed_charge_loss_min = []
    dynamics_computed_charge_loss_max = []
    figure_counter = 0
    for drop in drops:
        dynamics_computed_charge_loss_min.append(drop.dynamics_computed_delta_z_min)
        dynamics_computed_charge_loss_max.append(drop.dynamics_computed_delta_z_max)
        dropsMagnitude.append(float(drop.freq_change_magnitude))
        dropsCharge.append(float(drop.charge_change_magnitude))
        dropsSquaredRatioChange.append(float(drop.f_squared_ratio_change))
        C_loss_scaled_m_z.append(drop.C_loss_scaled_m_z)
        delta_m_z.append(drop.delta_m_z)
        delta_C_E.append(drop.delta_C_E)
        delta_C_E_percent.append(drop.delta_C_E_percent)
        scaled_delta_m_z.append(drop.scaled_m_z)
        delta_mass.append(drop.delta_mass)
        delta_charge.append(drop.freq_computed_charge_loss)

        dropsChargeChange.append(float(drop.delta_charge))
        freqComputedChargeLoss.append(float(drop.freq_computed_charge_loss))

    try:
        print("========= START ANALYSIS REPORT ==========")
        print("Average lifetime of stable traces: " + str(np.mean(no_drop_avg_lifetime)) + " steps")
        print("Average lifetime of all traces: " + str(np.mean(trace_lifetime)) + " steps")
        print("Average lifetime of all fragments: " + str(np.mean(fragment_lifetime)) + " steps")
        print("Selected data includes " + str(len(traces)) + " valid ions and " + str(
            len(drops)) + " recorded emission events.")
        print("Rejected ions based on slope: " + str(fail_count_slope))
        print("Rejected ions based on mass: " + str(fail_count_mass))
        print("Rejected ions based on charge: " + str(fail_count_charge))
        print("Rejected ions based on energy: " + str(fail_count_energy))
        print("Rejected ions based on f_computed_drop: " + str(fail_count_drop))
        print("Rejected ions based on noise: " + str(fail_count_noise))
        print("Traces with zero-length fragments: " + str(noncrit_zero_div_errors))
        print("========= END ANALYSIS REPORT ==========")

    except Exception:
        print('Missing some random variable in text output...')

    if export_drop_files:
        dbfile = open(str(analysis_name) + '_ion_object_files.pickle', 'wb')
        pickle.dump(drops, dbfile)
        dbfile.close()

    if dynamics_charge_loss == 1:
        dbfile = open(str(analysis_name) + '_dynamics_charge_loss_min.pickle', 'wb')
        pickle.dump(dynamics_computed_charge_loss_min, dbfile)
        dbfile.close()
        dbfile = open(str(analysis_name) + '_dynamics_charge_loss_max.pickle', 'wb')
        pickle.dump(dynamics_computed_charge_loss_max, dbfile)
        dbfile.close()
        if save_plots:
            Dynamics.plotter(fig_save_dir)

    if ion_frequencies:
        dbfile = open(str(analysis_name) + '_initial_ion_frequencies.pickle', 'wb')
        pickle.dump(initial_freq_collection, dbfile)
        dbfile.close()
        if save_plots:
            IonFrequencies.plotter(fig_save_dir)

    if drops_per_trace:
        dbfile = open(str(analysis_name) + '_drops_per_trace.pickle', 'wb')
        pickle.dump(drop_counts, dbfile)
        dbfile.close()
        if save_plots:
            DropsPerTrace.plotter(fig_save_dir)

    if trace_slope_distribution:
        dbfile = open(str(analysis_name) + '_slope_dist.pickle', 'wb')
        pickle.dump(included_slopes, dbfile)
        dbfile.close()
        if save_plots:
            TraceSlopeDist.plotter(fig_save_dir)

    if f_computed_charge_loss:
        dbfile = open(str(analysis_name) + '_amp_computed_charge_loss.pickle', 'wb')
        pickle.dump(dropsChargeChange, dbfile)
        dbfile.close()

        dbfile = open(str(analysis_name) + '_freq_computed_charge_loss.pickle', 'wb')
        pickle.dump(freqComputedChargeLoss, dbfile)
        dbfile.close()
        if save_plots:
            DropPlotter.EmissionPlotter(fig_save_dir)

    if C_E_percent_change:
        dbfile = open(str(analysis_name) + '_CE_percent_change.pickle', 'wb')
        pickle.dump(delta_C_E_percent, dbfile)
        dbfile.close()

    if HAR_eV_distribution:
        dbfile = open(str(analysis_name) + '_HAR_eV_aggregated.pickle', 'wb')
        pickle.dump([HAR_collection, energy_collection], dbfile)
        dbfile.close()
        if save_plots:
            EnergyPlotter.plotter(fig_save_dir)

    if mass_spectrum_2D:
        dbfile = open(str(analysis_name) + '_2D_mass_spectrum.pickle', 'wb')
        pickle.dump([mass_collection, charge_collection], dbfile)
        dbfile.close()
        if save_plots:
            MSPlotter_2D.MSPlotter(fig_save_dir)

    if mass_freq and save_plots:
        MassFreq.plotter(fig_save_dir)

    mass_collection_scaled = []
    for mass in mass_collection:
        mass_collection_scaled.append((mass / salt_MW) * salt_n)

    charge_collection_scaled = []
    for charge in charge_collection:
        charge_collection_scaled.append(charge * charge)

    if z_2_n:
        z2_n = []
        bins = []
        for n in range(len(mass_collection)):
            z2_n.append(charge_collection_scaled[n] / mass_collection_scaled[n])

        dbfile = open(str(analysis_name) + '_z2_n.pickle', 'wb')
        pickle.dump(z2_n, dbfile)
        dbfile.close()
