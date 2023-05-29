from enum import IntEnum, auto
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from preprocess import preprocess_centerline


class Feature(IntEnum):
    PEAK = auto()
    TROUGH = auto()


def find_extrema(der):
    is_pos = der >= 0
    sign_change = np.logical_xor(is_pos[1:], is_pos[:-1])
    intern_extrema = 1 + np.flatnonzero(sign_change)
    return intern_extrema


def sort_and_add_edges(ys, extrema, values, classif, min_width):
    argsort = sorted(range(len(extrema)), key=extrema.__getitem__)
    extrema = [extrema[i] for i in argsort]
    values = [values[i] for i in argsort]
    classif = [classif[i] for i in argsort]
    if extrema and extrema[0] >= min_width / 2:
        extrema.insert(0, 0)
        values.insert(0, ys[0])
    elif extrema:
        classif.pop(0)
    if extrema and len(ys) - 1 - extrema[-1] >= min_width / 2:
        extrema.append(len(ys) - 1)
        values.append(ys[-1])
    elif extrema:
        classif.pop()
    return extrema, values, classif


def resample_extrema(intern_extrema, ys, min_width):
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
            classif.append(Feature.PEAK)
        else:
            classif.append(Feature.TROUGH)
    return sort_and_add_edges(ys, extrema, values, classif, min_width)


def build_areas(ys, extrema, values):
    limits = []
    for (x_1, y_1, x_2, y_2) in zip(extrema, values, extrema[1:], values[1:]):
        if x_2 - x_1 <= 1.5:
            x = (x_1 + x_2) / 2
        else:
            mid = (y_1 + y_2) / 2
            i = math.ceil(x_1)
            try:
                while (ys[i + 1] - mid) * (y_1 - y_2) > 0:
                    i += 1
                x = i + (ys[i] - mid) / (y_1 - y_2)
            except IndexError:
                x = (x_1 + x_2) / 2
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


def filter_areas(ys, areas, min_width, min_depth, classif):
    while areas:
        widths = [r - l for l, r in areas]
        i_min = min(range(len(widths)), key=widths.__getitem__)
        width = widths[i_min]
        if width <= min_width or (width <= 3 * min_width and
                                  max_depth(ys, *areas[i_min]) < min_depth):
            areas, classif = remove_area(areas, classif, i_min)
        else:
            break
    return areas, classif


def find_peaks_troughs(xs, ys, kernel_len, std_cut, window, smooth_std,
                       min_width, min_depth, smoothing=True):
    if smoothing:
        xs, ys = preprocess_centerline(xs, ys, kernel_len, std_cut, window)
    ys_smooth = gaussian_filter1d(ys, smooth_std, mode="nearest")
    der = ys_smooth[1:] - ys_smooth[:-1]
    intern_extrema = find_extrema(der)
    extrema, values, classif = resample_extrema(intern_extrema, ys_smooth,
                                                min_width)
    areas = build_areas(ys_smooth, extrema, values)
    areas, classif = filter_areas(ys_smooth, areas, min_width, min_depth,
                                  classif)
    peaks = []
    troughs = []
    for area, feature in zip(areas, classif):
        area = np.array(area, dtype=np.float64) + xs[0]
        match feature:
            case Feature.PEAK:
                peaks.append(area)
            case Feature.TROUGH:
                troughs.append(area)
            case _:
                raise ValueError(f"Unknown feature {feature}, feature should "
                                 f"be a {Feature.PEAK} or a {Feature.TROUGH}.")
    return xs, ys, peaks, troughs
