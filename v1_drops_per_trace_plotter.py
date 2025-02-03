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
    files = generate_filelist(folder, '_drops_per_trace.pickle')
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

    drop_counts = []

    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for freq in db:
            drop_counts.append(freq)
        dbfile.close()

    active_ions = 0
    for ion in drop_counts:
        if ion > 0:
            active_ions += 1
    # try:
    #     print("Active ions: " + str(active_ions) + " (" + str(int((active_ions/len(drop_counts)) * 100)) + "%)")
    # except:
    #     print("Active ions: 0%")

    fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
    ax.hist(drop_counts, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8], align='left', color='black')
    labels = ["0", "1", "2", "3"]

    ax.set_title("")
    ax.set_xlabel('Emission Events per Single Trace', fontsize=24, weight='bold')
    ax.set_ylabel('Counts', fontsize=24, weight='bold')
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    plt.savefig(analysis_name + 'exported_drops_per_trace.png', bbox_inches='tight', dpi=300.0,
                transparent='true')


if __name__ == "__main__":
    folder = fd.askdirectory(title="Choose top folder")
    plotter(folder)