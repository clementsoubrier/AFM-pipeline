import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np


def plot_single_centerline(xs, ys, peaks, troughs):
    plt.plot(xs, ys, color="black")
    for x_1, x_2 in peaks:
        plt.axvspan(x_1, x_2, alpha=0.5, color="red")
    for x_1, x_2 in troughs:
        plt.axvspan(x_1, x_2, alpha=0.5, color="green")
    plt.show()


def plot_cell_centerlines(cell_centerlines, cell_peaks=None, cell_troughs=None,
                          v_offset=10, cell_id=None):
    offset = 0
    peaks_x = []
    peaks_y = []
    troughs_x = []
    troughs_y = []
    for centerline, peaks, troughs in zip(cell_centerlines, cell_peaks,
                                          cell_troughs):
        xs = centerline[:, 0]
        ys = centerline[:, 1] + offset
        if peaks.size:
            peaks_x.extend(peaks[:, 0])
            peaks_y.extend(peaks[:, 1] + offset)
        if troughs.size:
            troughs_x.extend(troughs[:, 0])
            troughs_y.extend(troughs[:, 1] + offset)
        plt.plot(xs, ys)
        offset += v_offset
    plt.scatter(peaks_x, peaks_y, c="red")
    plt.scatter(troughs_x, troughs_y, c="green")
    if cell_id is not None:
        plt.title(str(cell_id))
    plt.show()


def plot_3d_centerlines(cell_centerlines, cell_peaks=None, cell_troughs=None,
                        cell_id=None):
    width = max(map(len, cell_centerlines))
    shape = (len(cell_centerlines), width)
    xs_3d = np.zeros(shape, dtype=np.float64)
    ys_3d = np.zeros(shape, dtype=np.float64)
    zs_3d = np.zeros(shape, dtype=np.float64)
    ax = plt.axes(projection='3d')
    for i, centerline in enumerate(cell_centerlines):
        size = len(centerline)
        xs = centerline[:, 0]
        zs = centerline[:, 1]
        xs_3d[i, :size] = xs
        xs_3d[i, size:] = xs[-1]
        ys_3d[i, :] = i
        zs_3d[i, :size] = zs
        zs_3d[i, size:] = zs[-1]
        ax.plot3D(xs, [i] * size, zs + 0.1, 'black')
    ax.plot_surface(xs_3d, ys_3d, zs_3d, cmap="cividis", lw=0.5, rstride=1,
                    cstride=1, alpha=0.7, edgecolor='none',
                    norm=mplc.PowerNorm(gamma=0.6))
    if cell_peaks is not None:
        for i, peaks in enumerate(cell_peaks):
            if peaks.size:
                ax.scatter(peaks[:, 0], [i] * len(peaks), peaks[:, 1] + 0.1,
                           c="red", edgecolors='none', s=42)
    if cell_troughs is not None:
        for i, troughs in enumerate(cell_troughs):
            if troughs.size:
                ax.scatter(troughs[:, 0], [i] * len(troughs), troughs[:, 1] + 
                           0.1, c="green", edgecolors='none', s=42)
    if cell_id is not None:
        ax.set_title(str(cell_id))
    plt.show()
