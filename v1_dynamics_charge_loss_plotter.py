import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_filelist(folder, termString):
    # NOTE: folders variable must be a list, even if it is a list of one
    filelist = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(termString):
                filelist.append(os.path.join(root, file))
                # print("Loaded: " + filelist[-1])
    return filelist


def gauss(x, A, mu, sigma, offset):
    return offset + A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def integrator(x_array, y_array, x_min, x_max):
    x_min_index = find_nearest(x_array, x_min)
    x_max_index = find_nearest(x_array, x_max)
    try:
        integration = 0
        while x_min_index < x_max_index:
            integration = integration + y_array[x_min_index]
            x_min_index = x_min_index + 1
        print("Integration from " + str(x_min) + " to " + str(x_max) + ": " + str(integration))
        return integration

    except:
        print("Integration failed for unspecified reason.")


def plotter(folder):
    folder = folder.rsplit('.', maxsplit=1)[0] + ".tv7p"
    analysis_name = folder.rsplit('.', maxsplit=1)[0]
    new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]
    analysis_name = analysis_name + '.figures/'
    try:
        os.mkdir(analysis_name)
    except FileExistsError:
        # print("Path exists already.")
        junk = 0
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

    dynamics_charge_loss_min = []
    files = generate_filelist(folder, '_dynamics_charge_loss_min.pickle')
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for freq in db:
            dynamics_charge_loss_min.append(freq)
        dbfile.close()

    dynamics_charge_loss_max = []
    files = generate_filelist(folder, '_dynamics_charge_loss_max.pickle')
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for freq in db:
            dynamics_charge_loss_max.append(freq)
        dbfile.close()

    fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
    bin_count = 200  # 40 bins for 0.1 spacing on -4 to 0 plots
    hist_out = ax.hist(dynamics_charge_loss_min, bin_count, range=[-10, 0], color='forestgreen', alpha=1)
    hist_out = ax.hist(dynamics_charge_loss_max, bin_count, range=[-10, 0], color='red', alpha=0.25)
    # hist_out = ax.hist(dynamics_charge_loss_min, bin_count, range=[-40, 0], color='forestgreen')
    # hist_out = ax.hist(dynamics_charge_loss_max, bin_count, range=[-40, 0], color='red')
    ax.set_title("")
    ax.set_xlabel('Charge', fontsize=24, weight='bold')
    ax.set_ylabel('Counts', fontsize=24, weight='bold')
    ax.set_xticks([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0])
    # ax.set_xticks([-40, -36, -32, -28, -24, -20, -16, -12, -8, -4, 0])
    ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=3, length=4)
    ax.tick_params(axis='y', which='minor', width=3, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

    # plt.axvline(-1, color=fit_color, linestyle='solid', linewidth=3)
    # plt.axvline(-2, color=fit_color, linestyle='solid', linewidth=3)

    plt.savefig(analysis_name + 'exported_dynamics_computed.png', bbox_inches='tight', dpi=300.0, pad_inches=0.5,
                transparent='true')
