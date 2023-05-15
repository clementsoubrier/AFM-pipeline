from heapq import heapify, heappop

import numpy as np
from scipy.signal import convolve


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
    pad = kernel_len // 2
    kernel = [1 / kernel_len] * kernel_len
    ys = convolve(ys, kernel, mode="valid")
    return xs[pad: -pad], ys


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
    end = len(xs) - 2 - is_steep[: -window - 1: -1].argmax()
    while is_steep[end]:
        end -= 1
    return xs[start: end + 2], ys[start: end + 2]


def preprocess_centerline(xs, ys, kernel_len, std_cut, window):
    xs, ys = int_resample(xs, ys)
    xs, ys = smoothing(xs, ys, kernel_len)
    xs, ys = double_intersection(xs, ys)
    xs, ys = derivative_cut(xs, ys, std_cut, window)
    return xs, ys
