import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def MSPlotter(mass_collection, mass_axis_points, title_plt=""):
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

    fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
    histogram = ax.hist(mass_collection, 250, range=[mass_axis_points[0], mass_axis_points[-1]], color='black')
    peaks = find_peaks(histogram[0], width=3)
    print(histogram[1][peaks[0]]/1000000)

    ax.set_title(title_plt, fontsize=28, weight='bold')
    ax.set_xlabel('Mass (MDa)', fontsize=24, weight='bold')
    ax.set_ylabel('Counts', fontsize=24, weight='bold')
    mass_axis_labels = [str(x / 1000000) for x in mass_axis_points]
    ax.set_xticks(mass_axis_points, mass_axis_labels)
    ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.show()


if __name__ == "__main__":
    folder = fd.askdirectory(title="Choose top folder")
    plotter(folder)