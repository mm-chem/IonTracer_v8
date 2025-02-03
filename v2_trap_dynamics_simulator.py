import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import concentration_math as Rayleigh


class Initial_Ion:
    def __init__(self, mass, charge, eV_z):
        self.c_e = None
        self.mass = mass
        self.charge = charge
        self.m_z = mass / charge
        self.ke = None
        self.pe = None
        self.frequency = None
        self.position = None
        self.eV_z = eV_z
        self.fragments = []
        (self.pe_function, self.ke_function, self.pos_from_ke_function, self.pe_t_para_function,
         self.ke_t_para_function) = (self.import_energy_function(
            "/Users/mmcpartlan/Desktop/f_loss_simulations/energy_function.csv", self.eV_z))
        self.sample()

    def sample(self):
        self.compute_c_e()
        self.compute_location()
        self.compute_frequency()

    def compute_c_e(self, SPAMM=2):
        if SPAMM == 2:
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

        if SPAMM == 3:
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
            trap_V = 330

        self.c_e = (A00 + A10 * self.eV_z + A01 * trap_V + A20 * self.eV_z ** 2 + A11 * self.eV_z * trap_V + A02 *
                    trap_V ** 2 + A30 * self.eV_z ** 3 + A21 * self.eV_z ** 2 * trap_V + A12 * self.eV_z *
                    trap_V ** 2 + A03 * trap_V ** 3) ** 2

    def compute_location(self, fission_time=0.0):
        max_time = 183.0
        if fission_time == 0.0:
            fission_time = np.random.rand() * max_time
        self.pe = self.pe_t_para_function(fission_time)
        self.ke = self.ke_t_para_function(fission_time)
        self.position = self.pos_from_ke_function(self.ke)

    def compute_frequency(self):
        self.frequency = np.sqrt(self.c_e / self.m_z)

    @staticmethod
    def import_energy_function(path, max_energy):
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

        potential_energy = (np.array(potential_energy) / max(potential_energy)) * max_energy
        kinetic_energy = (np.array(kinetic_energy) / max(kinetic_energy)) * max_energy

        continuous_potential_e = interpolate.interp1d(ion_x_position, potential_energy, fill_value="extrapolate")
        continuous_kinetic_e = interpolate.interp1d(ion_x_position, kinetic_energy, fill_value="extrapolate")
        position_from_kinetic_e = interpolate.interp1d(kinetic_energy, ion_x_position, fill_value="extrapolate")
        continuous_potential_e_time_parameterized = interpolate.interp1d(ion_TOF, potential_energy,
                                                                         fill_value="extrapolate")
        continuous_kinetic_e_time_parameterized = interpolate.interp1d(ion_TOF, kinetic_energy,
                                                                       fill_value="extrapolate")

        return (continuous_potential_e, continuous_kinetic_e, position_from_kinetic_e,
                continuous_potential_e_time_parameterized, continuous_kinetic_e_time_parameterized)


class LocationEngine(Initial_Ion):
    def __init__(self, initial_ion, delta_mass, delta_charge, SPAMM):
        self.SPAMM = SPAMM
        super().__init__(initial_ion.mass, initial_ion.charge, initial_ion.eV_z)
        # Initialize with starting mass, charge, energy. Record deltas here...
        self.initial_mass = initial_ion.mass
        self.initial_charge = initial_ion.charge
        self.initial_eV_z = initial_ion.eV_z
        self.delta_mass = delta_mass
        self.delta_charge = delta_charge
        self.voltage_proportionality_k = None
        self.delta_eV_z = None
        self.delta_frequency = None
        self.f_computed_charge_loss = None
        # Sample with flight time at 0 and at 91.88 to get minimum and maximum bounds on delta-f for a given charge
        # loss event
        super().compute_location(fission_time=183.0)
        self.compute_event()
        self.min_freq_change = self.delta_frequency
        self.reset_to_parent()
        super().compute_location(fission_time=91.85)
        self.compute_event()
        self.max_freq_change = self.delta_frequency
        self.reset_to_parent()
        self.sample()
        self.compute_event()

    def update_fission(self, new_delta_charge, new_delta_mass):
        self.reset_to_parent()
        self.delta_charge = new_delta_charge
        self.delta_mass = new_delta_mass
        super().compute_location(fission_time=183.0)
        self.compute_event()
        self.min_freq_change = self.delta_frequency
        self.reset_to_parent()
        super().compute_location(fission_time=91.85)
        self.compute_event()
        self.max_freq_change = self.delta_frequency

        self.reset_to_parent()
        self.sample()
        self.compute_event()

    def location_simulator(self, iterations):
        delta_frequency_array = []
        f_computed_charge_loss_array = []
        delta_eV_z_array = []
        counter = 0
        while counter < iterations:
            self.compute_event()
            delta_frequency_array.append(self.delta_frequency)
            f_computed_charge_loss_array.append(self.f_computed_charge_loss)
            delta_eV_z_array.append(self.delta_eV_z)
            self.reset_to_parent()
            counter = counter + 1
        return delta_frequency_array, f_computed_charge_loss_array, delta_eV_z_array

    def reset_to_parent(self):
        self.mass = self.initial_mass
        self.charge = self.initial_charge
        self.eV_z = self.initial_eV_z
        self.m_z = self.mass / self.charge
        self.c_e = None
        self.ke = None
        self.pe = None
        self.frequency = None
        self.position = None
        self.fragments = []
        self.voltage_proportionality_k = None
        self.delta_eV_z = None
        self.delta_frequency = None
        self.f_computed_charge_loss = None
        # Use this to recompute where the event took place
        self.sample()

    def compute_event(self):
        self.voltage_proportionality_k = self.ke / (self.pe + self.ke)

        charge_change_percent = abs(self.delta_charge / self.charge)
        mass_change_percent = abs(self.delta_mass / self.mass)
        # eV/z increases with charge loss, decreases with mass loss
        energy_post_event = (self.eV_z + (-self.eV_z * mass_change_percent + self.eV_z * charge_change_percent) *
                             self.voltage_proportionality_k)
        energy_change_percent = energy_post_event / self.eV_z  # What percent is conserved

        # Update ion parameters to reflect the fission event that has taken place
        # Deltas here are FINAL - INITIAL
        initial_frequency = self.frequency
        initial_charge = self.charge
        self.mass = self.mass + self.delta_mass
        self.charge = self.charge + self.delta_charge
        self.m_z = self.mass / self.charge
        self.delta_eV_z = energy_post_event - self.eV_z
        self.eV_z = energy_post_event
        super().compute_c_e(self.SPAMM)
        super().compute_frequency()
        self.delta_frequency = self.frequency - initial_frequency

        f_squared_ratio_change = (initial_frequency ** 2) / (self.frequency ** 2)
        self.f_computed_charge_loss = (-(f_squared_ratio_change - 1) * initial_charge)


class ExportInterface:
    def __init__(self, ion_mass, ion_charge, ion_energy, fixed_delta_mass=None, charge_sweep=None, SPAMM=2):
        self.delta_mass = None
        self.primary_fragment = None
        self.initial_ion = Initial_Ion(ion_mass, ion_charge, ion_energy)

        if charge_sweep is None:
            charge_sweep = [1, 35]

        if fixed_delta_mass is None:
            self.delta_mass = -self.rayleigh_mass(charge_sweep[0])
            self.primary_fragment = LocationEngine(self.initial_ion, self.delta_mass, -charge_sweep[0], SPAMM)
        else:
            self.delta_mass = fixed_delta_mass

        self.primary_fragment = LocationEngine(self.initial_ion, self.delta_mass, -charge_sweep[0], SPAMM)
        self.possible_charge_losses = list(range(charge_sweep[0], charge_sweep[1]))
        self.min_abs_freq_changes = []
        self.max_abs_freq_changes = []
        for e in self.possible_charge_losses:
            if fixed_delta_mass is None:
                self.delta_mass = -self.rayleigh_mass(e)

            self.primary_fragment.update_fission(-e, self.delta_mass)
            # These appear to be mismatched because the min freq curve is more negative (larger amplitude)
            self.min_abs_freq_changes.append(self.primary_fragment.max_freq_change)
            self.max_abs_freq_changes.append(self.primary_fragment.min_freq_change)
        interp_charge_axis = [-x for x in self.possible_charge_losses]
        self.min_abs_freq_changes_continuous = interpolate.interp1d(self.min_abs_freq_changes,
                                                                    interp_charge_axis,
                                                                    fill_value="extrapolate")
        self.max_abs_freq_changes_continuous = interpolate.interp1d(self.max_abs_freq_changes,
                                                                    interp_charge_axis,
                                                                    fill_value="extrapolate")

    def min_abs_freq_change_curve(self, delta_freq):
        # No confusion... make sure this is negative
        # THIS RETURNS THE SMALLEST MAGNITUDE CHANGE
        delta_freq = -abs(delta_freq)
        delta_charge = self.min_abs_freq_changes_continuous(delta_freq)
        # delta_charge = -(min(range(len(self.min_abs_freq_changes)), key=lambda i: abs(self.min_abs_freq_changes[i] -
        #                                                                               delta_freq)) + 1)
        print("Dynamics min delta charge found: ", delta_charge, " for freq ", delta_freq, " and delta mass ",
              -self.rayleigh_mass(abs(delta_charge)), " droplet mass ", self.primary_fragment.initial_mass,
              " droplet charge ", self.primary_fragment.initial_charge)
        return delta_charge

    def max_abs_freq_change_curve(self, delta_freq):
        # No confusion... make sure this is negative
        # THIS RETURNS THE LARGEST MAGNITUDE CHANGE
        delta_freq = -abs(delta_freq)
        delta_charge = self.max_abs_freq_changes_continuous(delta_freq)
        # delta_charge = -(min(range(len(self.max_abs_freq_changes)), key=lambda i: abs(self.max_abs_freq_changes[i] -
        #                                                                               delta_freq)) + 1)
        print("Dynamics min delta charge found: ", delta_charge, " for freq ", delta_freq, " and delta mass ",
              -self.rayleigh_mass(abs(delta_charge)), " droplet mass ", self.primary_fragment.initial_mass,
              " droplet charge ", self.primary_fragment.initial_charge)
        return delta_charge

    @staticmethod
    def rayleigh_mass(charge):
        droplet = Rayleigh.rayleigh_calculator_charge_given(charge)
        return droplet.mass_daltons


if __name__ == "__main__":
    SPAMM = 2

    initial_mass = 3000000
    initial_charge = 190
    initial_energy = 220
    charge_change = -38
    mass_change = -100000

    export_interface_test = ExportInterface(initial_mass, initial_charge, initial_energy, fixed_delta_mass=None,
                                            charge_sweep=[1, 20])
    print("Test... charge loss (min bounded): ", export_interface_test.max_abs_freq_change_curve(-3500))

    plt.plot(range(1, len(export_interface_test.min_abs_freq_changes) + 1), export_interface_test.min_abs_freq_changes)
    plt.plot(range(1, len(export_interface_test.max_abs_freq_changes) + 1), export_interface_test.max_abs_freq_changes)
    # for e in range(1, len(export_interface_test.max_abs_freq_changes) + 1):
    #     plt.axvline(e, linestyle='dashdot', color='r')
    plt.show()

    test_ion = Initial_Ion(initial_mass, initial_charge, initial_energy)
    primary_fragment = LocationEngine(test_ion, mass_change, charge_change, SPAMM)
    print("Primary fragment initial frequency: ", int(primary_fragment.frequency))
    primary_delta_freq, primary_f_computed, primary_eV_z = primary_fragment.location_simulator(10000)

    secondary_fragment = LocationEngine(test_ion, abs(mass_change) - initial_mass, abs(charge_change) - initial_charge,
                                        SPAMM)
    secondary_delta_freq, secondary_f_computed, secondary_eV_z = secondary_fragment.location_simulator(10000)

    plt.hist(primary_delta_freq, bins=100, range=[-4000, 0])
    plt.title("Primary Fragment: Delta Freq")
    plt.xlabel("Delta Freq (Hz)")
    plt.ylabel("Counts")
    plt.show()

    plt.hist(primary_eV_z, bins=100, range=[-20, 20])
    plt.title("Primary Fragment: Delta eV/z")
    plt.xlabel("Delta Energy (eV/z)")
    plt.ylabel("Counts")
    plt.show()

    plt.hist(secondary_delta_freq, bins=100, range=[0, 120000])
    plt.title("Secondary Fragment: Delta Freq")
    plt.xlabel("Delta Freq (Hz)")
    plt.ylabel("Counts")
    plt.show()

    plt.hist(secondary_eV_z, bins=100, range=[-50, 50])
    plt.title("Secondary Fragment: Delta eV/z")
    plt.xlabel("Delta Energy (eV/z)")
    plt.ylabel("Counts")
    plt.show()
