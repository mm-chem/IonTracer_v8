import math
import pickle
import numpy as np
from scipy import signal as sig
from scipy import stats
import matplotlib.pyplot as plt
import os
from tkinter import filedialog as fd
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter


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

    return m_list, qlist


def generate_filelist(folder, termString):
    # NOTE: folders variable must be a list, even if it is a list of one
    filelist = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(termString):
                filelist.append(os.path.join(root, file))
                # print("Loaded: " + filelist[-1])
    return filelist


def plotter(folder):
    folder = folder.rsplit('.', maxsplit=1)[0] + ".tv7p"
    files = generate_filelist(folder, '_mass_spectrum.pickle')
    analysis_name = folder.rsplit('.', maxsplit=1)[0]
    new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]
    analysis_name = analysis_name + '.figures/'
    try:
        os.mkdir(analysis_name)
    except FileExistsError:
        print("Path exists already.")
    analysis_name = analysis_name + '/' + new_folder_name

    SMALL_SIZE = 18
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    mass_collection = []

    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for mass in db[0]:
            mass_collection.append(mass)
        dbfile.close()

    files = generate_filelist(folder, '_initial_ion_frequencies.pickle')

    frequencies = []
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for point in db:
            frequencies.append(point)
        dbfile.close()

    # Plot a 2D mass spectrum
    # Rayleigh line parameters are in DIAMETER (nm)
    fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
    heatmap, xedges, yedges = np.histogram2d(frequencies, mass_collection, bins=[160, 120],
                                             range=[[8000, 30000], [0, 12000000]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    gaussmap = gaussian_filter(heatmap, 1, mode='nearest')

    # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
    ax.imshow(gaussmap.T, cmap='nipy_spectral_r', extent=extent, origin='lower', aspect='auto',
              interpolation='none')

    ax.set_title("")
    ax.set_xlabel('Initial Frequency (kHz)', fontsize=24, weight='bold')
    ax.set_ylabel('Mass (MDa)', fontsize=24, weight='bold')
    ax.set_xticks([8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000], ["8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "30"])
    ax.set_yticks([0, 2000000, 4000000, 6000000, 8000000, 10000000, 12000000], ["0", "2", "4", "6", "8", "10", "12"])
    ax.tick_params(axis='x', which='major', labelsize=26, width=3, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=3, length=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='minor', width=2, length=4)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.savefig(analysis_name + 'exported_mass_freq_2D.png', bbox_inches='tight', dpi=300.0, pad_inches=0.5,
                transparent='true')

    # Plot a ZOOMED 2D mass spectrum
    # Rayleigh line parameters are in DIAMETER (nm)
    fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
    heatmap, xedges, yedges = np.histogram2d(frequencies, mass_collection, bins=[160, 120],
                                             range=[[30000, 60000], [0, 120000]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    gaussmap = gaussian_filter(heatmap, 1, mode='nearest')

    # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
    ax.imshow(gaussmap.T, cmap='nipy_spectral_r', extent=extent, origin='lower', aspect='auto',
              interpolation='none')

    ax.set_title("")
    ax.set_xlabel('Initial Frequency (kHz)', fontsize=24, weight='bold')
    ax.set_ylabel('Mass (kDa)', fontsize=24, weight='bold')
    ax.set_xticks([30000, 35000, 40000, 45000, 50000, 55000, 60000],
                  ["30", "35", "40", "45", "50", "55", "60"])
    ax.set_yticks([0, 20000, 40000, 60000, 80000, 100000, 120000], ["0", "20", "40", "60", "80", "100", "120"])
    ax.tick_params(axis='x', which='major', labelsize=26, width=3, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=3, length=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='minor', width=2, length=4)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.savefig(analysis_name + 'exported_mass_freq_2D.png', bbox_inches='tight', dpi=300.0, pad_inches=0.5,
                transparent='true')


if __name__ == "__main__":
    folder = fd.askdirectory(title="Choose top folder")
    plotter(folder)
