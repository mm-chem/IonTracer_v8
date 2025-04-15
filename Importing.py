import csv
import numpy as np

"""
!!!!!!!!!!!!!!!!!!!!!!!!!
!DO NOT MODIFY THIS FILE!
!!!!!!!!!!!!!!!!!!!!!!!!!
"""

class Ion:
    def __init__(self,
                 file,
                 n_segments,
                 f_slope,
                 f_intercept,
                 f_fit_st_dev,
                 d,
                 d_st_dev,
                 e,
                 e_st_dev,
                 m_z,
                 m_z_st_dev,
                 z,
                 z_st_dev,
                 m,
                 m_st_dev):
        self.file = file
        self.n_segments = n_segments
        self.f_slope = f_slope
        self.f_intercept = f_intercept
        self.f_fit_st_dev = f_fit_st_dev
        self.d = d
        self.d_st_dev = d_st_dev
        self.e = e
        self.e_st_dev = e_st_dev
        self.m_z = m_z
        self.m_z_st_dev = m_z_st_dev
        self.z = z
        self.z_st_dev = z_st_dev
        self.m = m
        self.m_st_dev = m_st_dev

        self.filt = True
        self.sel = True

class Ion_Properties:
    def __init__(self,
                 origin,
                 filenames,
                 n_segments,
                 f_slope,
                 f_intercept,
                 f_fit_st_dev,
                 d,
                 d_st_dev,
                 e,
                 e_st_dev,
                 m_z,
                 m_z_st_dev,
                 z,
                 z_st_dev,
                 m,
                 m_st_dev):


        self.origin = origin
        self.counts = len(filenames)

        self.ions = []

        current_file = ""
        for i in range(0, len(filenames)):
            if filenames[i].strip() != "":
                current_file = filenames[i]
            
            self.ions.append(Ion(current_file,
                                 n_segments[i],
                                 f_slope[i],
                                 f_intercept[i],
                                 f_fit_st_dev[i],
                                 d[i],
                                 d_st_dev[i],
                                 e[i],
                                 e_st_dev[i],
                                 m_z[i],
                                 m_z_st_dev[i],
                                 z[i],
                                 z_st_dev[i],
                                 m[i],
                                 m_st_dev[i]))

    def __str__(self) -> str:
        rep = f"Object: {self.__class__.__name__}\n"
        rep += f"Data loaded from: {self.origin}\n"
        rep += f"Total ion counts: {self.counts}"
        return rep

    def _mask(self, key: str, mask: list | np.ndarray):
        if len(mask) == len(self.ions):
            for i in range(0, len(self.ions)):
                self.ions[i].__dict__[key] = mask[i]
                
    def _get(self, key, filt, sel):
        items = []
        for i in range(0, len(self.ions)):
            if filt or sel:
                if filt and sel:
                    if self.ions[i].__dict__["filt"] and self.ions[i].__dict__["sel"]:
                        items.append(self.ions[i].__dict__[key])        
                    else:
                        items.append(np.nan)

                elif filt and not sel:
                    if self.ions[i].__dict__["filt"]:
                        items.append(self.ions[i].__dict__[key])
                    else:
                        items.append(np.nan)
 
                elif not filt and sel:
                    if self.ions[i].__dict__["sel"]:
                        items.append(self.ions[i].__dict__[key])        
                    else:
                        items.append(np.nan)

            else:
                items.append(self.ions[i].__dict__[key])
        
        return np.asarray(items)

    def append(self, file):
        filenames = []
        n_segments = []
        frequency_slope = []
        frequency_intercept = []
        frequency_fit_st_dev = []
        duty_cycle = []
        duty_cycle_st_dev = []
        energy = []
        energy_st_dev = []
        m_z = []
        m_z_st_dev = []
        z = []
        z_st_dev = []
        m = []
        m_st_dev = []

        with open(file, newline="") as fo:
            reader = csv.reader(fo, delimiter=",")
            for row in reader:
                if "Data layout" in row[0]:
                    continue
                if len(row) == 15:
                    try:
                        filenames.append(row[0])
                        n_segments.append(int(row[1]))
                        frequency_slope.append(float(row[2]))
                        frequency_intercept.append(float(row[3]))
                        frequency_fit_st_dev.append(float(row[4]))
                        duty_cycle.append(float(row[5]))
                        duty_cycle_st_dev.append(float(row[6]))
                        energy.append(float(row[7]))
                        energy_st_dev.append(float(row[8]))
                        m_z.append(float(row[9]))
                        m_z_st_dev.append(float(row[10]))
                        z.append(float(row[11]))
                        z_st_dev.append(float(row[12]))
                        m.append(float(row[13]))
                        m_st_dev.append(float(row[14]))
                    except ValueError or TypeError:
                        pass

        filenames = np.asarray(filenames)
        n_segments = np.asarray(n_segments)
        f_slope = np.asarray(frequency_slope)
        f_intercept = np.asarray(frequency_intercept)
        f_fit_st_dev = np.asarray(frequency_fit_st_dev)
        d = np.asarray(duty_cycle)
        d_st_dev = np.asarray(duty_cycle_st_dev)
        e = np.asarray(energy)
        e_st_dev = np.asarray(energy_st_dev)
        m_z = np.asarray(m_z)
        m_z_st_dev = np.asarray(m_z_st_dev)
        z = np.asarray(z)
        z_st_dev = np.asarray(z_st_dev)
        m = np.asarray(m)
        m_st_dev = np.asarray(m_st_dev)

        current_file = ""
        for i in range(0, len(filenames)):
            if filenames[i].strip() != "":
                current_file = filenames[i]

            self.ions.append(Ion(current_file,
                                 n_segments[i],
                                 f_slope[i],
                                 f_intercept[i],
                                 f_fit_st_dev[i],
                                 d[i],
                                 d_st_dev[i],
                                 e[i],
                                 e_st_dev[i],
                                 m_z[i],
                                 m_z_st_dev[i],
                                 z[i],
                                 z_st_dev[i],
                                 m[i],
                                 m_st_dev[i]))


    def filter(self, mask: list | np.ndarray):
        self._mask("filt", mask)

    def select(self, mask: list | np.ndarray):
        self._mask("sel", mask)

    def get_n_segments(self, filt=False, sel=False):
        return self._get("n_segments", filt, sel)
    
    def get_f_slope(self, filt=False, sel=False):
        return self._get("f_slope", filt, sel)

    def get_f_intercept(self, filt=False, sel=False):
        return self._get("f_intercept", filt, sel)
    
    def get_f_fit_st_dev(self, filt=False, sel=False):
        return self._get("f_fit_st_dev", filt, sel)
    
    def get_d(self, filt=False, sel=False):
        return self._get("d", filt, sel)
        
    def get_d_st_dev(self, filt=False, sel=False):
        return self._get("d_st_dev", filt, sel)

    def get_e(self, filt=False, sel=False):
        return self._get("e", filt, sel)

    def get_e_st_dev(self, filt=False, sel=False):
        return self._get("e_st_dev", filt, sel)
    
    def get_m_z(self, filt=False, sel=False):
        return self._get("m_z", filt, sel)
    
    def get_m_z_st_dev(self, filt=False, sel=False):
        return self._get("m_z_st_dev", filt, sel)
    
    def get_z(self, filt=False, sel=False):
        return self._get("z", filt, sel)
    
    def get_z_st_dev(self, filt=False, sel=False):
        return self._get("z_st_dev", filt, sel)
    
    def get_m(self, filt=False, sel=False):
        return self._get("m", filt, sel)
    
    def get_m_st_dev(self, filt=False, sel=False):
        return self._get("m_st_dev", filt, sel)


def Read_Ion_Properties(file: str):
    filenames = []
    n_segments = []
    frequency_slope = []
    frequency_intercept = []
    frequency_fit_st_dev = []
    duty_cycle = []
    duty_cycle_st_dev = []
    energy = []
    energy_st_dev = []
    m_z = []
    m_z_st_dev = []
    z = []
    z_st_dev = []
    m = []
    m_st_dev = []

    with open(file, newline="") as fo:
        reader = csv.reader(fo, delimiter=",")
        for row in reader:
            if "Data layout" in row[0]:
                continue
            if len(row) == 15:
                try:
                    filenames.append(row[0])
                    n_segments.append(int(row[1])) 
                    frequency_slope.append(float(row[2]))
                    frequency_intercept.append(float(row[3]))
                    frequency_fit_st_dev.append(float(row[4]))
                    duty_cycle.append(float(row[5]))
                    duty_cycle_st_dev.append(float(row[6]))
                    energy.append(float(row[7]))
                    energy_st_dev.append(float(row[8]))
                    m_z.append(float(row[9]))
                    m_z_st_dev.append(float(row[10]))
                    z.append(float(row[11]))
                    z_st_dev.append(float(row[12]))
                    m.append(float(row[13]))
                    m_st_dev.append(float(row[14]))
                except ValueError or TypeError:
                    pass

    filenames = np.asarray(filenames)
    n_segments = np.asarray(n_segments)
    frequency_slope = np.asarray(frequency_slope)
    frequency_intercept = np.asarray(frequency_intercept)
    frequency_fit_st_dev = np.asarray(frequency_fit_st_dev)
    duty_cycle = np.asarray(duty_cycle)
    duty_cycle_st_dev = np.asarray(duty_cycle_st_dev)
    energy = np.asarray(energy)
    energy_st_dev = np.asarray(energy_st_dev)
    m_z = np.asarray(m_z)
    m_z_st_dev = np.asarray(m_z_st_dev)
    z = np.asarray(z)
    z_st_dev = np.asarray(z_st_dev)
    m = np.asarray(m)
    m_st_dev = np.asarray(m_st_dev)

    return Ion_Properties(file,
                          filenames,
                          n_segments,
                          frequency_slope,
                          frequency_intercept,
                          frequency_fit_st_dev,
                          duty_cycle,
                          duty_cycle_st_dev,
                          energy,
                          energy_st_dev,
                          m_z,
                          m_z_st_dev,
                          z,
                          z_st_dev,
                          m,
                          m_st_dev)
