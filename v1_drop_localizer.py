import math
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle
from scipy import interpolate
from scipy.optimize import minimize


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


def import_energy_function(path):
    potential_energy = []
    kinetic_energy = []
    ion_x_position = []
    ion_TOF = []
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            row = row[0].split(';')
            potential_energy.append(float(row[0]))
            kinetic_energy.append(float(row[1]))
            ion_x_position.append(float(row[2]))
            ion_TOF.append(float(row[3]))
    return potential_energy, kinetic_energy, ion_x_position, ion_TOF


def c_e_calculator(eV_z):
    trap_V = 336.1  # trapping cones potential
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

    c_e = (A00 + A10 * eV_z + A01 * trap_V + A20 * eV_z ** 2 + A11 * eV_z * trap_V + A02 * trap_V ** 2 + A30 * eV_z
           ** 3 + A21 * eV_z ** 2 * trap_V + A12 * eV_z * trap_V ** 2 + A03 * trap_V ** 3) ** 2
    return c_e


class ChargeLossObject:
    def __init__(self, charge_loss, mass_loss_mu, mass_loss_sigma):
        self.charge_loss = charge_loss
        self.mass_loss_mu = mass_loss_mu
        self.mass_loss_sigma = mass_loss_sigma
        self.randomized_mass_lost = None


class IonSpectrum:
    def __init__(self):
        # Uncomment for LaCl3
        # self.mass_collection = []
        # self.charge_collection = []
        # file = '/Users/mmcpartlan/Desktop/f_loss_simulations/composite_LaCl3_2D_mass_spectrum.pickle'
        # dbfile = open(file, 'rb')
        # db = pickle.load(dbfile)
        # for mass in db[0]:
        #     self.mass_collection.append(mass)
        # for charge in db[1]:
        #     self.charge_collection.append(charge)
        # dbfile.close()
        #
        # self.experimental_f_computed_charge_loss = []
        # file = "/Users/mmcpartlan/Desktop/f_loss_simulations/composite_LaCl3_freq_computed_charge_loss.pickle"
        # dbfile = open(file, 'rb')
        # db = pickle.load(dbfile)
        # for freq in db:
        #     self.experimental_f_computed_charge_loss.append(freq)
        # dbfile.close()

        # Uncomment for TMAA Data (from disparate files)
        self.mass_collection = []
        self.charge_collection = []
        files = ['/Users/mmcpartlan/Desktop/f_loss_simulations/TMAA_2D/TMAA_composite_2D_mass_spectrum1.pickle',
                 '/Users/mmcpartlan/Desktop/f_loss_simulations/TMAA_2D/TMAA_composite_2D_mass_spectrum2.pickle',
                 '/Users/mmcpartlan/Desktop/f_loss_simulations/TMAA_2D/TMAA_composite_2D_mass_spectrum3.pickle']
        for file in files:
            dbfile = open(file, 'rb')
            db = pickle.load(dbfile)
            for mass in db[0]:
                self.mass_collection.append(mass)
            for charge in db[1]:
                self.charge_collection.append(charge)
            dbfile.close()

        self.experimental_f_computed_charge_loss = []
        files = ['/Users/mmcpartlan/Desktop/f_loss_simulations/TMAA_losses/TMAA_composite_freq_computed_charge_loss1.pickle',
                 '/Users/mmcpartlan/Desktop/f_loss_simulations/TMAA_losses/TMAA_composite_freq_computed_charge_loss2.pickle',
                 '/Users/mmcpartlan/Desktop/f_loss_simulations/TMAA_losses/TMAA_composite_freq_computed_charge_loss3.pickle']
        for file in files:
            dbfile = open(file, 'rb')
            db = pickle.load(dbfile)
            for freq in db:
                self.experimental_f_computed_charge_loss.append(freq)
            dbfile.close()

        # Uncomment for randomly generated ions
        # mu_mass_1, sigma_mass_1, counts_1 = 1000000, 10000, 100000  # mean and standard deviation
        # mass_array_simulated_1 = np.random.normal(mu_mass_1, sigma_mass_1, counts_1)
        #
        # mu_charge_1, sigma_charge_1 = 200, 0  # mean and standard deviation
        # charge_array_simulated_1 = np.random.normal(mu_charge_1, sigma_charge_1, counts_1)
        #
        # mu_mass_2, sigma_mass_2, counts_2 = 1000000, 10000, 100000  # mean and standard deviation
        # mass_array_simulated_2 = np.random.normal(mu_mass_2, sigma_mass_2, counts_2)
        #
        # mu_charge_2, sigma_charge_2 = 200, 0  # mean and standard deviation
        # charge_array_simulated_2 = np.random.normal(mu_charge_2, sigma_charge_2, counts_2)
        #
        # mass_array_simulated = abs(np.concatenate((np.array(mass_array_simulated_1), np.array(mass_array_simulated_2))))
        # charge_array_simulated = abs(np.concatenate((np.array(charge_array_simulated_1), np.array(
        #     charge_array_simulated_2))))
        #
        # self.mass_collection = mass_array_simulated
        # self.charge_collection = charge_array_simulated

        # Uncomment for homogeneous ions
        # self.mass_collection = np.full(10000, 4000000)
        # self.charge_collection = np.full(10000, 150)

def freq_computed_optimizer(ion_mass_array, ion_charge_array, charge_loss_objects,
                            experimental_f_computed_charge_loss, percent_1, percent_2, percent_3, plot=True):
    # Basic ion properties
    m_z_array_simulated = []
    c_e_array_simulated = []
    initial_freq_array_simulated = []

    # Charge loss properties
    fission_time_simulated = []
    field_strength_at_event_simulated = []
    ke_energy_at_event_simulated = []
    frequency_loss_simulated = []
    f_computed_charge_loss_simulated = []
    voltage_proportionality_k_simulated = []
    energy_change_percent_simulated = []
    c_e_post_event_simulated = []
    location_of_event = []

    # charge_loss_events = np.random.choice([1, 2], counts)
    counts = len(ion_mass_array)
    charge_loss_events = np.random.choice(charge_loss_objects, counts, p=[percent_1, percent_2, percent_3])
    # charge_loss_events = np.random.choice(charge_loss_objects, counts)

    counts_1 = 0
    counts_2 = 0
    counts_3 = 0
    counts_4 = 0
    for ion in range(len(ion_mass_array)):
        if charge_loss_events[ion].charge_loss == 1:
            charge_loss_events[ion].randomized_mass_lost = np.random.normal(
                    charge_loss_events[ion].mass_loss_mu, charge_loss_events[ion].mass_loss_sigma)
            while charge_loss_events[ion].randomized_mass_lost < 0:
                charge_loss_events[ion].randomized_mass_lost = np.random.normal(
                    charge_loss_events[ion].mass_loss_mu, charge_loss_events[ion].mass_loss_sigma)
            counts_1 += 1
        if charge_loss_events[ion].charge_loss == 2:
            charge_loss_events[ion].randomized_mass_lost = np.random.normal(
                    charge_loss_events[ion].mass_loss_mu, charge_loss_events[ion].mass_loss_sigma)
            while charge_loss_events[ion].randomized_mass_lost < 0:
                charge_loss_events[ion].randomized_mass_lost = np.random.normal(
                    charge_loss_events[ion].mass_loss_mu, charge_loss_events[ion].mass_loss_sigma)
            counts_2 += 1
        if charge_loss_events[ion].charge_loss == 3:
            charge_loss_events[ion].randomized_mass_lost = np.random.normal(
                charge_loss_events[ion].mass_loss_mu, charge_loss_events[ion].mass_loss_sigma)
            while charge_loss_events[ion].randomized_mass_lost < 0:
                charge_loss_events[ion].randomized_mass_lost = np.random.normal(
                    charge_loss_events[ion].mass_loss_mu, charge_loss_events[ion].mass_loss_sigma)
            counts_3 += 1
        if charge_loss_events[ion].charge_loss == 4:
            charge_loss_events[ion].randomized_mass_lost = np.random.normal(
                charge_loss_events[ion].mass_loss_mu, charge_loss_events[ion].mass_loss_sigma)
            while charge_loss_events[ion].randomized_mass_lost < 0:
                charge_loss_events[ion].randomized_mass_lost = np.random.normal(
                    charge_loss_events[ion].mass_loss_mu, charge_loss_events[ion].mass_loss_sigma)
            counts_4 += 1

    if plot:
        print("+1 loss events:" + str(counts_1))
        print("+2 loss events:" + str(counts_2))
        print("+3 loss events:" + str(counts_3))
        print("+4 loss events:" + str(counts_4))

    mu_eVz, sigma_eVz = 220, 2  # mean and standard deviation
    eV_per_z_array_simulated = np.random.normal(mu_eVz, sigma_eVz, counts)

    # Build the energy function from Conner's data
    potential_energy, kinetic_energy, ion_x_position, ion_TOF = import_energy_function(
        "/Users/mmcpartlan/Desktop/f_loss_simulations/energy_function.csv")
    continuous_potential_e = interpolate.interp1d(ion_x_position, potential_energy, fill_value="extrapolate")
    continuous_kinetic_e = interpolate.interp1d(ion_x_position, kinetic_energy, fill_value="extrapolate")
    position_from_kinetic_e = interpolate.interp1d(kinetic_energy, ion_x_position, fill_value="extrapolate")
    continuous_potential_e_time_parameterized = interpolate.interp1d(ion_TOF, potential_energy,
                                                                     fill_value="extrapolate")
    continuous_kinetic_e_time_parameterized = interpolate.interp1d(ion_TOF, kinetic_energy,
                                                                   fill_value="extrapolate")

    # Verify that the function was built correctly
    # t_test = np.linspace(0.0, 183, 1000)
    # plt.plot(t_test, continuous_kinetic_e_time_parameterized(t_test))
    # plt.plot(t_test, (continuous_potential_e_time_parameterized(t_test)))
    # plt.xlabel('Time in Trap (us)')
    # plt.ylabel('Experienced Potential (V)')
    # plt.show()

    # Start simulating emission events... time parameterized
    max_time = 183.0
    for ion in range(len(ion_mass_array)):
        m_z_array_simulated.append(ion_mass_array[ion] / ion_charge_array[ion])
        c_e_array_simulated.append(c_e_calculator(eV_per_z_array_simulated[ion]))
        initial_freq_array_simulated.append(np.sqrt(c_e_array_simulated[ion] / m_z_array_simulated[ion]))

        fission_time = np.random.rand() * max_time
        pe_at_event = continuous_potential_e_time_parameterized(fission_time)
        ke_at_event = continuous_kinetic_e_time_parameterized(fission_time)
        voltage_proportionality_k = ke_at_event / (pe_at_event + ke_at_event)
        voltage_proportionality_k_simulated.append(voltage_proportionality_k)

        fission_time_simulated.append(fission_time)
        field_strength_at_event_simulated.append(pe_at_event)
        ke_energy_at_event_simulated.append(ke_at_event)
        location_of_event.append(position_from_kinetic_e(ke_at_event))

        mass_change_percent = charge_loss_events[ion].randomized_mass_lost / ion_mass_array[ion]
        charge_change_percent = charge_loss_events[ion].charge_loss / ion_charge_array[ion]
        # eV/z increases with charge loss, decreases with mass loss
        energy_post_event = eV_per_z_array_simulated[ion] + (-eV_per_z_array_simulated[ion] * mass_change_percent +
                                                             eV_per_z_array_simulated[ion] *
                                                             charge_change_percent) * voltage_proportionality_k

        energy_change_percent = energy_post_event / eV_per_z_array_simulated[ion]  # What percent is conserved
        energy_change_percent_simulated.append(energy_change_percent)

        m_z_post_event = ((ion_mass_array[ion] - charge_loss_events[ion].randomized_mass_lost) /
                          (ion_charge_array[ion] - charge_loss_events[ion].charge_loss))
        # freq_post_event = np.sqrt(c_e_array_simulated[ion] / m_z_post_event)
        c_e_post_event = c_e_calculator(energy_post_event)
        c_e_post_event_simulated.append(c_e_post_event)
        freq_post_event = np.sqrt(c_e_post_event / m_z_post_event)
        # frequency_loss_simulated.append((initial_freq_array_simulated[ion] - freq_post_event) * np.random.uniform(0.7,
        #                                                                                                           1.3))
        frequency_loss_simulated.append(initial_freq_array_simulated[ion] - freq_post_event)
        f_squared_ratio_change = (initial_freq_array_simulated[ion] ** 2) / (freq_post_event ** 2)
        f_computed_charge_loss_simulated.append(-(f_squared_ratio_change - 1) * ion_charge_array[ion])

    # Plot initial frequencies (calcualted from eV/z in trap)
    # plt.hist(initial_freq_array_simulated, bins=100, color='maroon')
    # plt.xlabel('Ion Initial Frequency (Hz)')
    # plt.ylabel('Counts')
    # plt.show()

    # Plot simulated fission times (randomly selected)
    # plt.hist(fission_time_simulated, bins=100, color='maroon')
    # plt.xlabel('Ion Fission Time (us)')
    # plt.ylabel('Counts')
    # plt.show()

    # Plot simulated energy % changes
    # plt.hist(energy_change_percent_simulated, bins=100, color='maroon')
    # plt.xlabel('% of Energy Conserved Over Fission')
    # plt.ylabel('Counts')
    # plt.show()

    # Plot c(e) after fission
    # plt.hist(c_e_post_event_simulated, bins=100, color='maroon')
    # plt.xlabel('C(E) Term After Fission')
    # plt.ylabel('Counts')
    # plt.show()

    # Plot simulated ke/total values at fission times (from time parameterized potential equation)
    # plt.hist(voltage_proportionality_k_simulated, bins=100, color='maroon')
    # plt.xlabel('KE/Total E Ratio at Fission (us)')
    # plt.ylabel('Counts')
    # plt.show()

    # Plot simulated trap potential values at fission times (from time parameterized potential equation)
    # plt.hist(field_strength_at_event_simulated, bins=100, color='maroon')
    # plt.xlabel('KE/Total E Ratio at Fission (us)')
    # plt.ylabel('Counts')
    # plt.show()

    # Plot delta frequencies
    # plt.hist(frequency_loss_simulated, bins=100, color='maroon')
    # plt.xlabel('Frequency Lost (Hz)')
    # plt.ylabel('Counts')
    # plt.show()

    # Plot m/z of simulated ions
    # plt.hist(m_z_array_simulated, bins=100, color='maroon')
    # plt.xlabel('Mass to Charge (m/z)')
    # plt.ylabel('Counts')
    # plt.show()

    # ==================================================================================
    # Handling experimental data
    # ==================================================================================

    counts_exp, bins_exp = np.histogram(experimental_f_computed_charge_loss, bins=100, range=[-6, 0])
    exp_normalized_counts = (counts_exp / (np.max(counts_exp)))
    exp_bins_truncated = bins_exp[:-1]
    if plot:
        plt.hist(exp_bins_truncated, bins_exp, weights=exp_normalized_counts, color='black', alpha=0.6)

    # Plot freq computed charge loss of simulated ions
    counts_sim, bins_sim = np.histogram(f_computed_charge_loss_simulated, bins=100, range=[-6, 0])
    sim_normalized_counts = (counts_sim / (np.max(counts_sim)))
    sim_bins_truncated = bins_sim[:-1]
    if plot:
        plt.hist(sim_bins_truncated, bins_sim, weights=sim_normalized_counts, color='maroon', alpha=0.6)
        plt.xlabel('Frequency Computed Charge Loss (e)')
        plt.ylabel('Counts')

    fit_score = sum(abs(exp_normalized_counts - sim_normalized_counts))
    fit_score_weights = abs(exp_normalized_counts - sim_normalized_counts)
    # plt.hist(sim_bins_truncated, bins_sim, weights=fit_score_weights, color='orange', alpha=0.6)
    if plot:
        plt.show()

    # ==================================================================================
    # Handling experimental data  ^^^^^^^^^^^^^^^^^
    # ==================================================================================

    # Plot a 2D mass spectrum
    # Rayleigh line parameters are in DIAMETER (nm)
    # fig, ax = plt.subplots(layout='tight', figsize=(14, 7))
    # rayleigh_x, rayleigh_y = plot_rayleigh_line(axis_range=[0, 200])
    # ax.plot(rayleigh_x, rayleigh_y, color='black', linestyle="dashed", linewidth=2)
    # heatmap, xedges, yedges = np.histogram2d(ion_mass_array, ion_charge_array, bins=[160, 120],
    #                                          range=[[0, 12000000], [0, 500]])
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # gaussmap = gaussian_filter(heatmap, 1, mode='nearest')
    #
    # # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
    # ax.imshow(gaussmap.T, cmap='nipy_spectral_r', extent=extent, origin='lower', aspect='auto',
    #           interpolation='none')
    #
    # ax.set_title("")
    # ax.set_xlabel('Mass (MDa)', fontsize=24, weight='bold')
    # ax.set_ylabel('Charge', fontsize=24, weight='bold')
    # ax.set_xticks([0, 2000000, 4000000, 6000000, 8000000, 10000000, 12000000], ["0", "2", "4", "6", "8", "10", "12"])
    # # ax.set_yticks(hist_charge_bins, hist_charge_labels)
    # ax.tick_params(axis='x', which='major', labelsize=26, width=3, length=8)
    # ax.tick_params(axis='y', which='major', labelsize=26, width=3, length=8)
    # ax.minorticks_on()
    # ax.tick_params(axis='x', which='minor', width=2, length=4)
    # ax.tick_params(axis='y', which='minor', width=2, length=4)
    # ax.spines['bottom'].set_linewidth(3)
    # ax.spines['left'].set_linewidth(3)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_linewidth(3)
    # ax.spines['top'].set_linewidth(3)
    # plt.show()

    return fit_score


class OptoPackager:
    def __init__(self):
        self.ion_spectrum = IonSpectrum()

    def opto_packager(self, X, plot=False):
        ML_mu1, ML_sig1, ML_mu2, ML_sig2, ML_mu3, ML_sig3, percent_1, percent_2, percent_3 = X
        # masses, charges = generate_random_test_ions()
        loss_object_1 = ChargeLossObject(1, ML_mu1, ML_sig1)
        loss_object_2 = ChargeLossObject(2, ML_mu2, ML_sig2)
        loss_object_3 = ChargeLossObject(3, ML_mu3, ML_sig3)
        object_array = [loss_object_1, loss_object_2, loss_object_3]
        fit_score = freq_computed_optimizer(self.ion_spectrum.mass_collection, self.ion_spectrum.charge_collection,
                                            object_array, self.ion_spectrum.experimental_f_computed_charge_loss,
                                            percent_1, percent_2, percent_3, plot=plot)
        print(fit_score)
        return fit_score

    def optimize_drops(self, iterations):
        ML1_min_mass = 500
        ML1_max_mass = 1500

        ML2_min_mass = 5000
        ML2_max_mass = 20000

        ML3_min_mass = 15000
        ML3_max_mass = 30000

        best_result = 1000000
        # ML1, ML1_sigma, ML2, ML2_sigma, percent_+1_loss
        best_fit_params = [None, None, None, None, None, None, None]
        for i in range(iterations):
            ML1_mass = np.random.uniform(ML1_min_mass, ML1_max_mass)
            ML1_sigma = (np.random.uniform(1, 33) / 100) * ML1_mass
            ML2_mass = np.random.uniform(ML2_min_mass, ML2_max_mass)
            ML2_sigma = (np.random.uniform(1, 33) / 100) * ML2_mass
            ML3_mass = np.random.uniform(ML3_min_mass, ML3_max_mass)
            ML3_sigma = (np.random.uniform(1, 33) / 100) * ML3_mass
            while ML1_mass > ML2_mass:
                ML1_mass = np.random.uniform(ML1_min_mass, ML1_max_mass)
                ML1_sigma = (np.random.uniform(1, 33) / 100) * ML1_mass
                ML2_mass = np.random.uniform(ML2_min_mass, ML2_max_mass)
                ML2_sigma = (np.random.uniform(1, 33) / 100) * ML2_mass
                ML3_mass = np.random.uniform(ML3_min_mass, ML3_max_mass)
                ML3_sigma = (np.random.uniform(1, 33) / 100) * ML3_mass

            percent_1 = 1
            percent_2 = 0
            percent_3 = 0
            while percent_1 >= 0:
                X = [ML1_mass, ML1_sigma, ML2_mass, ML2_sigma, ML3_mass, ML3_sigma, percent_1, percent_2, percent_3]
                print(X)
                result = self.opto_packager(X)
                if result < best_result:
                    best_result = result
                    best_fit_params = [ML1_mass, ML1_sigma, ML2_mass, ML2_sigma, ML3_mass, ML3_sigma, percent_1,
                                       percent_2, percent_3]
                percent_1 = percent_1 - 0.1
                percent_2 = percent_2 + 0.1
                print('Best Result: ' + str(best_result))

        return best_result, best_fit_params

    def plot_with_set_values(self, params):
        ML1_mass, ML1_sigma, ML2_mass, ML2_sigma, ML3_mass, ML3_sigma, percent_1, percent_2, percent_3 = params
        X = [ML1_mass, ML1_sigma, ML2_mass, ML2_sigma, ML3_mass, ML3_sigma, percent_1, percent_2, percent_3]
        result = self.opto_packager(X, plot=True)
        print(result)


if __name__ == "__main__":
    optimizer = OptoPackager()
    # res, params = optimizer.optimize_drops(100)
    # print(res)
    # print(params)
    # params = ML1_mass, ML1_sigma, ML2_mass, ML2_sigma, percent_1
    # params = 0.0005, 0.00001, 0.002, 0.0005, 0.5
    # params = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    # optimizer.plot_with_set_values(params)
    # params = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    # optimizer.plot_with_set_values(params)
    # params = [3800, 190, 18500, 300, 35000, 3000, 0.45, 0.3, 0.25]
    # params = [0, 0, 18500, 300, 35000, 3000, 1, 0, 0]
    # params = [3800, 380, 18500, 370, 30000, 3000, 0.5, 0.3, 0.2]  # Optimal for LaCl3 (verify later)
    params = [1000, 500, 12000, 6000, 0, 0, 0.3, 0.7, 0]
    optimizer.plot_with_set_values(params)
