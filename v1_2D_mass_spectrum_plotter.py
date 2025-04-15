import math
import numpy as np
import matplotlib.pyplot as plt
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


def MSPlotter(mass_collection, charge_collection, mass_axis_points, charge_axis_range, title_plt=""):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 36

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Plot a 2D mass spectrum
    # Rayleigh line parameters are in DIAMETER (nm)
    fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
    rayleigh_x, rayleigh_y = plot_rayleigh_line(axis_range=[0, 200])
    ax.plot(rayleigh_x, rayleigh_y, color='black', linestyle="dashed", linewidth=2)
    heatmap, xedges, yedges = np.histogram2d(mass_collection, charge_collection, bins=[160, 120],
                                             range=[[mass_axis_points[0], mass_axis_points[-1]], [charge_axis_range[0], charge_axis_range[-1]]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    gaussmap = gaussian_filter(heatmap, 1, mode='nearest')

    # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
    ax.imshow(gaussmap.T, cmap='nipy_spectral_r', extent=extent, origin='lower', aspect='auto',
              interpolation='none')

    ax.set_title(title_plt, fontsize=28, weight='bold')
    ax.set_xlabel('Mass (MDa)', fontsize=24, weight='bold')
    ax.set_ylabel('Charge', fontsize=24, weight='bold')
    mass_axis_labels = [str(x / 1000000) for x in mass_axis_points]
    ax.set_xticks(mass_axis_points, mass_axis_labels)
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

    plt.show()
