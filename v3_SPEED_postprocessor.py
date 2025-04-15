import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog as fd
import v3_SPEED_analysis as STFT



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
    print("ANALYSIS PERFORMED FOR SPEED INSTRUMENT")
    print("---------------------------------------")
    print(str(SPAMM))
    print("---------------------------------------")

