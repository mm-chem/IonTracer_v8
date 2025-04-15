import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter


def get_1d_histogram(x: list[float] | np.ndarray,
                     x_lim: list[float, float] | tuple[float, float],
                     x_bin_size: float,
                     smooth: list[float, float] | tuple[float, float] = None,
                     normalize: bool = False,
                     plottable_as_line: bool = True):
    x_lim_new = [*x_lim]
    rem = (x_lim_new[1]-x_lim_new[0]) % x_bin_size
    if rem != 0:
        x_lim_new[1] += x_bin_size - (x_lim_new[1]-x_lim_new[0]) % x_bin_size

    if smooth is not None:
        edge_width = smooth[0]*2*x_bin_size
        x_lim_new[0] -= edge_width
        x_lim_new[1] += edge_width

    n_x_bins = int((x_lim_new[1]-x_lim_new[0]) / x_bin_size)

    bins, edges = np.histogram(x, range=x_lim_new, bins=n_x_bins)

    if plottable_as_line:
        ret_x = (edges - (edges[1] - edges[0])/2)[1:]
    else:
        ret_x = edges

    if smooth is not None:
        ret_y = savgol_filter(bins, smooth[0], smooth[1])
    else:
        ret_y = bins

    if normalize:
        ret_y = ret_y / ret_y.max()

    if smooth is not None:
        edge_width = smooth[0] * 2
        return ret_x[edge_width:(len(ret_x)-edge_width)], ret_y[edge_width:(len(ret_y)-edge_width)]
    else:
        return ret_x, ret_y


def get_2d_histogram(x: list[float] | np.ndarray,
                     x_lim: list[float, float] | tuple[float, float],
                     x_bin_size: float,
                     y: list[float] | np.ndarray,
                     y_lim: list[float, float] | tuple[float, float],
                     y_bin_size: float,
                     smooth: list[float, float, float, float] | tuple[float, float, float, float] = None,
                     normalize: bool = False):

    x_lim_new = [*x_lim]
    x_lim_new[1] += x_bin_size - (x_lim_new[1] - x_lim_new[0]) % x_bin_size
    n_x_bins = int((x_lim_new[1] - x_lim_new[0]) / x_bin_size)

    y_lim_new = [*y_lim]
    y_lim_new[1] += y_bin_size - (y_lim_new[1] - y_lim_new[0]) % y_bin_size
    n_y_bins = int((y_lim_new[1] - y_lim_new[0]) / y_bin_size)

    H, x_edges, y_edges = np.histogram2d(x, y, range=[x_lim_new, y_lim_new], bins=[n_x_bins, n_y_bins])

    ret_x, ret_y = np.meshgrid(x_edges, y_edges)
    if smooth is not None:
        ret_z = gaussian_filter(H.T, sigma=[smooth[0], smooth[1]], order=[smooth[2], smooth[3]])
    else:
        ret_z = H.T

    if normalize:
        return ret_x, ret_y, ret_z/ret_z.max()
    else:
        return ret_x, ret_y, ret_z




