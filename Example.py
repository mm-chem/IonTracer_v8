import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from Importing import Read_Ion_Properties
import v1_2D_mass_spectrum_plotter as plotter_2d
import v1_1D_MS_plotter as plotter_1d
import math

def main():

    # DegP and mNEON complexes formed immediately after mixing (out of PBS buffer)
    # files = ["/Users/mmcpartlan/Desktop/DegP Studies/DegP_mNEON_mix_dil10x_15min_PBS/Acquisition 1 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/DegP_mNEON_mix_dil10x_15min_PBS/Acquisition 2 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/DegP_mNEON_mix_dil10x_15min_PBS/Acquisition 3 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/DegP_mNEON_mix_dil10x_15min_PBS/Acquisition 4 - analysis "
    #          "new/ion_properties.txt"]
    # plot_title = "DegP + mNEON (Mixed in PBS, Sprayed in PBS)"

    # DegP and mNEON complexes formed after mixing in PBS, THEN buffer exchanging into AA
    # files = ["/Users/mmcpartlan/Desktop/DegP Studies/DegP_mNEON_AA_mixed_in_PBS/Acquisition 3 - analysis "
    #          "new/ion_properties.txt"]
    # plot_title = "DegP + mNEON (Mixed in PBS, Sprayed in AA)"
    #
    # # Run with this to see DegP in AA before mixing with mNEON client
    # files = ["/Users/mmcpartlan/Desktop/DegP Studies/DegP_AA/Acquisition 1 - analysis new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/DegP_AA/Acquisition 2 - analysis new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/DegP_AA/Acquisition 3 - analysis new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/DegP_AA/Acquisition 4 - analysis new/ion_properties.txt"]
    # plot_title = "DegP in AA"

    # Run with these files to see DegP and mNEON complexes formed immediately after mixing (post buffer exchange)
    # files = ["/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_mixed_in_AA/Acquisition 1 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_mixed_in_AA/Acquisition 2 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_mixed_in_AA/Acquisition 3 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_mixed_in_AA/Acquisition 4 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_mixed_in_AA/Acquisition 5 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_mixed_in_AA/Acquisition 6 - analysis "
    #          "new/ion_properties.txt"]
    # plot_title = "DegP + mNEON (Mixed in AA, Sprayed in AA)"

    # Run with these files to see DegP and mNEON complexes formed 24h after mixing (post buffer exchange)
    # files = ["/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_mixed_after_exchange_day2/Acquisition 6 - analysis "
    #          "new/ion_properties.txt"]
    # plot_title = "DegP + mNEON (Mixed in AA, Sprayed in AA, 24h)"

    # DegP and mNEON complexes formed 24h after mixing (out of PBS buffer)
    # files = ["/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_PBS_10x_dil_day2/Acquisition 6 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_PBS_10x_dil_day2/Acquisition 7 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_PBS_10x_dil_day2/Acquisition 8 - analysis "
    #          "new/ion_properties.txt",
    #          "/Users/mmcpartlan/Desktop/DegP Studies/degP_mNEON_PBS_10x_dil_day2/Acquisition 9 - analysis "
    #          "new/ion_properties.txt"]
    # plot_title = "DegP + mNEON (Mixed in PBS, Sprayed in PBS, 24h)"

    # Filepath for 100nm nanospheres data
    files = ["/Users/mmcpartlan/Desktop/SPAMM 3 Data/04-04/100nm_nanospheres/Acquisition 1 - analysis "
             "new/ion_properties.txt"]
    plot_title = "100nm Thermo Nanospheres (1% Acid)"

    props = []
    for file in files:
        props.append(Read_Ion_Properties(file))
        n_segments = props[-1].get_n_segments()
        f_fit_stdev = props[-1].get_f_fit_st_dev()
        mask = []
        for i in range(0, props[-1].counts):
            if f_fit_stdev[i] > 5 or n_segments[i] < 5:
                mask.append(False)
            else:
                mask.append(True)
        props[-1].filter(mask)

    mass_collection = []
    charge_collection = []
    energy_collection = []
    for prop in props:
        charge_collection.extend(prop.get_z(filt=True))
        mass_collection.extend(prop.get_m(filt=True))

    plotter_2d.MSPlotter(mass_collection, charge_collection, [0, 100000000, 200000000, 300000000, 400000000, 500000000,
                                                              600000000], [0, 1500], title_plt=plot_title)

    plotter_1d.MSPlotter(mass_collection, [0, 1000000, 2000000, 3000000, 4000000, 5000000,
                                                              6000000], title_plt=plot_title)

    plotter_1d.MSPlotter(mass_collection, [0, 250000, 500000, 750000, 1000000, 1250000,
                                           1500000], title_plt=plot_title)

if __name__ == "__main__":
    main()