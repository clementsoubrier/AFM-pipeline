from heapq import heapify, heappop

import numpy as np
from scipy.signal import convolve


REF_PIXEL_SIZE = 0.07874015748031496  # micrometers
REF_VERTI_SCALE = 1


def get_scaled_parameters(pixel_size, verti_scale=None, preprocessing=True,
                          peaks_troughs=False, filtering=False, misc=False,
                          ref_pixel_size=REF_PIXEL_SIZE,
                          ref_verti_scale=REF_VERTI_SCALE, kernel_len=3,
                          std_cut=2.5, window=3, smooth_std=1.5, min_width=4,
                          min_depth=1.5, min_len=40, min_prep_len=35,
                          max_der_std=5, max_der=15, max_translation=15,
                          v_offset=10):
    h_scaling = ref_pixel_size / pixel_size
    if verti_scale is None:
        v_scaling = 1
    else:
        v_scaling = ref_verti_scale / verti_scale
    params = {}
    if preprocessing:
        params["kernel_len"] = 1 + round(h_scaling * (kernel_len - 1))
        params["std_cut"] = std_cut
        params["window"] = 1 + round(h_scaling * (window - 1))
    if peaks_troughs:
        params["smooth_std"] = h_scaling * smooth_std
        params["min_width"] = h_scaling * min_width
        params["min_depth"] = v_scaling * min_depth
    if filtering:
        params["min_len"] = h_scaling * min_len
        params["min_prep_len"] = h_scaling * min_prep_len
        params["max_der_std"] = max_der_std
        params["max_der"] = (v_scaling / h_scaling) * max_der
    if misc:
        params["max_translation"] = max(1, round(h_scaling * max_translation))
        params["v_offset"] = v_scaling * v_offset
    return params


def int_resample(xs, ys):
    xs = xs - xs[0]
    int_xs = [0]
    int_ys = [ys[0]]
    x = 0
    i = 1
    while True:
        x += 1
        if x > xs[i]:
            i += 1
            if i == len(xs):
                break
        int_xs.append(x)
        a = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        int_ys.append(a * ys[i] + (1 - a) * ys[i - 1])
    int_xs = np.array(int_xs, dtype=np.float64)
    int_ys = np.array(int_ys, dtype=np.float64)
    return int_xs, int_ys 


def smoothing(xs, ys, kernel_len):
    kernel = [1 / kernel_len] * kernel_len
    xs = convolve(xs, kernel, mode="valid")
    ys = convolve(ys, kernel, mode="valid")
    return xs, ys


def double_intersection(xs, ys):
    if ys[-1] < ys[0]:
        xs, ys = double_intersection(xs[:: -1], ys[:: -1])
        return xs[:: -1], ys[:: -1]
    heap = list(ys)
    heapify(heap)
    i = 0
    try:
        while heappop(heap) == ys[i]:
            i += 1
    except IndexError:
        return xs, ys
    if i:
        i -= 1
    return xs[i:], ys[i:]


def derivative_cut(xs, ys, std_cut, window):
    deriv = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    mean = np.mean(deriv)
    std = np.std(deriv)
    is_steep = abs(deriv - mean) >= std_cut * std
    start = is_steep[:window].argmax()
    while is_steep[start]:
        start += 1
        if start == len(ys):
            return xs, ys
    end = len(xs) - 2 - is_steep[: -window - 1: -1].argmax()
    while is_steep[end]:
        end -= 1
        if end == -1:
            return xs, ys
    return xs[start: end + 2], ys[start: end + 2]


def preprocess_centerline(xs, ys, kernel_len, std_cut, window):
    xs, ys = int_resample(xs, ys)
    xs, ys = smoothing(xs, ys, kernel_len)
    xs, ys = double_intersection(xs, ys)
    xs, ys = derivative_cut(xs, ys, std_cut, window)
    return xs, ys


def keep_centerline(xs, ys, min_len, kernel_len, std_cut, window, min_prep_len,
                    max_der_std, max_der):
    if xs[-1] - xs[0] < min_len:
        return np.bool_(False)
    corrupted = np.logical_or(ys <= 2, ys >= 253)
    if 40 * np.count_nonzero(corrupted) >= len(ys):
        return np.bool_(False)
    if np.ptp(ys) <= 2.55e-10:
        return np.bool_(False)
    xs_p, _ = preprocess_centerline(xs, ys, kernel_len, std_cut, window)
    if xs_p[-1] - xs_p[0] < min_prep_len:
        return np.bool_(False)
    start = np.searchsorted(xs, xs_p[0], "right") - 1
    end = np.searchsorted(xs, xs_p[-1], "left") + 1
    dx = xs[start + 1: end] - xs[start: end - 1]
    dy = ys[start + 1: end] - ys[start: end - 1]
    der = dy / dx
    mean = np.mean(der)
    std = np.std(der)
    abnormal_variation = abs(der - mean) >= min(max_der_std * std, max_der)
    if 40 * np.count_nonzero(abnormal_variation) >= len(ys):
        return np.bool_(False)
    return np.bool_(True)
