import math

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from align import align_centerlines
from group_by_cell import get_centerlines_by_cell
from preprocess import preprocess_centerline


PEAK = 0
TROUGH = 1

PLOT_NONE = 0
PLOT_CENTERLINE = 1
PLOT_CELL = 2
PLOT_3D = 3


def find_extrema(der):
    is_pos = der >= 0
    sign_change = np.logical_xor(is_pos[1:], is_pos[:-1])
    intern_extrema = 1 + np.flatnonzero(sign_change)
    return intern_extrema


def sort_and_add_edges(ys, extrema, values, classif):
    argsort = sorted(range(len(extrema)), key=extrema.__getitem__)
    extrema = [extrema[i] for i in argsort]
    values = [values[i] for i in argsort]
    classif = [classif[i] for i in argsort]
    if extrema and extrema[0] >= 2:
        extrema.insert(0, 0)
        values.insert(0, ys[0])
    elif extrema:
        classif.pop(0)
    if extrema and len(ys) - 1 - extrema[-1] >= 2:
        extrema.append(len(ys) - 1)
        values.append(ys[-1])
    elif extrema:
        classif.pop()
    return extrema, values, classif


def resample_extrema(intern_extrema, ys):
    extrema = []
    values = []
    classif = []
    for i in intern_extrema:
        a = (ys[i + 1] - 2 * ys[i] + ys[i - 1]) / 2
        b = (ys[i + 1] - ys[i - 1]) / 2
        c = ys[i]
        x = i - b / (2 * a)
        y = c - b ** 2 / (4 * a)
        extrema.append(x)
        values.append(y)
        if a < 0:
            classif.append(PEAK)
        else:
            classif.append(TROUGH)
    return sort_and_add_edges(ys, extrema, values, classif)


def build_areas(ys, extrema, values):
    limits = []
    for (x_1, y_1, x_2, y_2) in zip(extrema, values, extrema[1:], values[1:]):
        if x_2 - x_1 <= 1.5:
            x = (x_1 + x_2) / 2
        else:
            mid = (y_1 + y_2) / 2
            i = math.ceil(x_1)
            while (ys[i + 1] - mid) * (y_1 - y_2) > 0:
                i += 1
            x = i + (ys[i] - mid) / (y_1 - y_2)
        limits.append(x)
    areas = list(zip(limits, limits[1:]))
    return areas


def max_depth(ys, x_l, x_r):
    i_l = math.floor(x_l)
    a_l = x_l - i_l
    y_l = a_l * ys[i_l + 1] + (1 - a_l) * ys[i_l]
    i_r = math.ceil(x_r)
    a_r = i_r - x_r
    y_r = a_r * ys[i_r - 1] + (1 - a_r) * ys[i_r]
    dx = x_r - x_l
    dy = y_r - y_l
    norm = math.hypot(dx, dy)
    dx /= norm
    dy /= norm
    max_dist = 0
    for i in range(i_l + 1, i_r):
        dist = abs(dx * (y_l - ys[i]) - dy * (x_l - i))
        max_dist = max(max_dist, dist)
    return max_dist
    

def remove_area(areas, classif, i):
    if i == 0:
        return areas[1:], classif[1:]
    if i == len(areas) - 1:
        return areas[: -1], classif[: -1]
    area = (areas[i - 1][0], areas[i + 1][1])
    areas = areas[: i - 1] + [area] + areas[i + 2:]
    classif = classif[: i - 1] + classif[i + 1:]
    return areas, classif


def filter_areas(ys, areas, min_depth, classif):
    while areas:
        widths = [r - l for l, r in areas]
        i_min = min(range(len(widths)), key=widths.__getitem__)
        width = widths[i_min]
        if width <= 4 or (width <= 12 and
                          max_depth(ys, *areas[i_min]) < min_depth):
            areas, classif = remove_area(areas, classif, i_min)
        else:
            break
    return areas, classif


def find_peaks_troughs(xs, ys, kernel_len, std_cut, window, min_depth,
                       smoothing=True):
    if smoothing:
        xs, ys = preprocess_centerline(xs, ys, kernel_len, std_cut, window)
    ys_smooth = gaussian_filter1d(ys, 1.5, mode="nearest")
    der = ys_smooth[1:] - ys_smooth[:-1]
    intern_extrema = find_extrema(der)
    extrema, values, classif = resample_extrema(intern_extrema, ys_smooth)
    areas = build_areas(ys_smooth, extrema, values)
    areas, classif = filter_areas(ys_smooth, areas, min_depth, classif)
    peaks = []
    troughs = []
    for area, feature in zip(areas, classif):
        area = np.array(area, dtype=np.float64) + xs[0]
        if feature == PEAK:
            peaks.append(area)
        elif feature == TROUGH:
            troughs.append(area)
        else:
            raise ValueError(f"Unknown feature {feature}, feature should be a"
                             f"peak {PEAK} or a trough {TROUGH}.")
    return xs, ys, peaks, troughs


def areas_to_points(cell_areas, cell_centerlines, feature):
    cell_points = []
    if feature == PEAK:
        extremum = max
    elif feature == TROUGH:
        extremum = min
    else:
        raise ValueError(f"Unknown feature {feature}, feature should "
                            f"be a peak {PEAK} or a trough {TROUGH}.")
    for areas, centerline in zip(cell_areas, cell_centerlines):
        x_0 = centerline[0, 0]
        ys = centerline[:, 1]
        points = []
        for l, r in areas:
            i = math.ceil(l - x_0)
            j = math.floor(r - x_0)
            k = extremum(range(i, j + 1), key=ys.__getitem__)
            x = k + x_0
            y = ys[k]
            points.append((x, y))
        cell_points.append(np.array(points, dtype=np.float64))
    return cell_points


def plot_single_centerline(xs, ys, peaks, troughs):
    plt.plot(xs, ys, color="black")
    for x_1, x_2 in peaks:
        plt.axvspan(x_1, x_2, alpha=0.5, color="red")
    for x_1, x_2 in troughs:
        plt.axvspan(x_1, x_2, alpha=0.5, color="green")
    plt.show()


def plot_cell_centerlines(cell_centerlines, cell_peaks, cell_troughs,
                          v_offset, cell_id):
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
    plt.title(str(cell_id))
    plt.show()


def plot_3d_centerlines(cell_centerlines, cell_peaks, cell_troughs):
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
    for i, peaks in enumerate(cell_peaks):
        if peaks.size:
            ax.scatter(peaks[:, 0], [i] * len(peaks), peaks[:, 1] + 0.1,
                       c="red", edgecolors='none', s=42)
    for i, troughs in enumerate(cell_troughs):
        if troughs.size:
            ax.scatter(troughs[:, 0], [i] * len(troughs), troughs[:, 1] + 0.1,
                       c="green", edgecolors='none', s=42)
    plt.show()


def main():
    kernel_len = 3
    std_cut = 2.5
    window = 3
    min_depth = 1.5
    v_offset = 10

    dataset = "05-02-2014"
    plot_mode = PLOT_CENTERLINE

    for cell, cell_id in get_centerlines_by_cell(dataset):
        cell_centerlines = [preprocess_centerline(xs, ys, kernel_len, std_cut,
                                                  window) for xs, ys in cell]
        cell_centerlines = align_centerlines(*cell_centerlines)
        cell_peaks = []
        cell_troughs = []
        for centerline in cell_centerlines:
            xs = centerline[:, 0]
            ys = centerline[:, 1]
            xs, ys, peaks, troughs = find_peaks_troughs(xs, ys, kernel_len,
                                                        std_cut, window,
                                                        min_depth, False)
            if plot_mode == PLOT_CENTERLINE:
                plot_single_centerline(xs, ys, peaks, troughs)
            cell_peaks.append(peaks)
            cell_troughs.append(troughs)

        cell_peaks = areas_to_points(cell_peaks, cell_centerlines, PEAK)
        cell_troughs = areas_to_points(cell_troughs, cell_centerlines, TROUGH)
        if plot_mode == PLOT_CELL:
            plot_cell_centerlines(cell_centerlines, cell_peaks, cell_troughs,
                                  v_offset, cell_id)
        if plot_mode == PLOT_3D and len(cell_centerlines) > 1:
            plot_3d_centerlines(cell_centerlines, cell_peaks, cell_troughs)
        

if __name__ == "__main__":
    main()
