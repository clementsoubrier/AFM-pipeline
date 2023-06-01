import numpy as np


def align_max_covariance(prev, ys, max_translation):
    if len(ys) > len(prev):
        return -align_max_covariance(ys, prev, max_translation)
    best_cov = -np.inf
    best_delta = None
    min_delta = -max_translation
    max_delta = len(prev) + max_translation - len(ys)
    for delta in range(min_delta, max_delta + 1):
        a = prev[max(0, delta): min(len(prev), delta + len(ys))]
        b = ys[max(0, -delta): len(ys) - max(0, delta + len(ys) - len(prev))]
        cov = np.dot(a - np.mean(a), b - np.mean(b))
        if cov > best_cov:
            best_cov = cov
            best_delta = delta
    return best_delta


def align_min_l2(prev, ys, max_translation, penalty=0.25):
    if len(ys) > len(prev):
        return -align_min_l2(ys, prev, max_translation, penalty)
    best_err = np.inf
    best_delta = None
    min_delta = -max_translation
    max_delta = len(prev) + max_translation - len(ys)
    for delta in range(min_delta, max_delta + 1):
        a = prev[max(0, delta): min(len(prev), delta + len(ys))]
        b = ys[max(0, -delta): len(ys) - max(0, delta + len(ys) - len(prev))]
        diff = (a - np.mean(a)) - (b - np.mean(b))
        stick_out = max(0, len(b) + delta - len(a)) - min(0, delta)
        dist_err = np.sqrt(np.dot(diff, diff) / len(a))
        stick_out_err = penalty * stick_out / np.sqrt(len(a))
        err = dist_err + stick_out_err
        if err < best_err:
            best_err = err
            best_delta = delta
    return best_delta


def align_centerlines(*centerlines, max_translation=None):
    xs = []
    ys = []
    for x, y in centerlines:
        xs.append(x - x[0])
        ys.append(y - np.median(y))
    prev_x_0 = 0
    prev = ys[0]
    aligned = []
    for x, y in zip(xs, ys):
        if max_translation is None:
            max_translation = round((len(prev) + len(y)) / 6)
        delta = align_min_l2(prev, y, max_translation)
        x += delta + prev_x_0
        prev_x_0 = x[0]
        prev = y
        aligned.append(np.column_stack((x, y)))
    return aligned
