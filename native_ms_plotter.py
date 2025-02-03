import csv
import matplotlib.pyplot as plt
import os
from tkinter import filedialog as fd


def generate_filelist(folder, termString):
    # NOTE: folders variable must be a list, even if it is a list of one
    filelist = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(termString):
                filelist.append(os.path.join(root, file))
                print("Loaded: " + filelist[-1])
    return filelist


if __name__ == "__main__":
    SMALL_SIZE = 24
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 28

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    folder = fd.askdirectory(title="Choose top folder")
    folder = folder.rsplit('.', maxsplit=1)[0] + ".spec_raw"
    files = generate_filelist(folder, '.csv')
    analysis_name = folder.rsplit('.', maxsplit=1)[0]
    new_folder_name = analysis_name.rsplit('/', maxsplit=1)[-1]
    analysis_name = analysis_name + '.spec/'
    try:
        os.mkdir(analysis_name)
    except FileExistsError:
        print("Path exists already.")
    analysis_name = analysis_name + '/' + new_folder_name

    axis_2d = []
    data_2d = []

    for file in files:
        axis_1d = []
        data_1d = []
        with open(file, mode='r') as spectrum:
            csvFile = csv.reader(spectrum)
            for lines in csvFile:
                try:
                    x = float(lines[0].split(';')[0])
                    y = float(lines[0].split(';')[1])
                    axis_1d.append(x)
                    data_1d.append(y)
                except:
                    print('Deleted point...')
        axis_2d.append(axis_1d)
        data_2d.append(data_1d)

    for i in range(len(axis_2d)):
        print('Generating plot ' + str(i))
        fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
        ax.plot(axis_2d[i], data_2d[i], color='maroon', linewidth=5)

        ax.set_title("")
        ax.set_xlabel('m/z', fontsize=36, weight='bold')
        ax.set_ylabel('Intensity', fontsize=36, weight='bold')
        ax.tick_params(axis='x', which='major', labelsize=32, width=5, length=8)
        ax.tick_params(axis='y', which='major', labelsize=32, width=5, length=8)
        ax.ticklabel_format(style='plain')
        ax.set_yticklabels([])  # Hide y-intensity values
        ax.set_yticks([])  # Hide y-intensity ticks
        ax.set_ylabel('Intensity', fontsize=36, weight='bold', labelpad=25)  # Add some space between axis and label
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)

        plt.savefig(analysis_name + '_spec_' + str(i) + '.png', bbox_inches='tight', dpi=300.0,
                    transparent='true')
        print('Finished plot ' + str(i))
