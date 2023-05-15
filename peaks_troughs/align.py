import numpy as np


def align_max_covariance(prev, ys):
    if len(ys) > len(prev):
        return -align_max_covariance(ys, prev)
    best_cov = -np.inf
    best_delta = None
    for delta in range(-15, len(prev) + 16 - len(ys)):
        a = prev[max(0, delta): min(len(prev), delta + len(ys))]
        b = ys[max(0, -delta): len(ys) - max(0, delta + len(ys) - len(prev))]
        cov = np.dot(a - np.mean(a), b - np.mean(b))
        if cov > best_cov:
            best_cov = cov
            best_delta = delta
    return best_delta


def align_centerlines(*centerlines):
    xs = []
    ys = []
    for x, y in centerlines:
        xs.append(x - x[0])
        ys.append(y - np.median(y))
    prev_x_0 = 0
    prev = ys[0]
    aligned = []
    for x, y in zip(xs, ys):
        delta = align_max_covariance(prev, y)
        x += delta + prev_x_0
        prev_x_0 = x[0]
        prev = y
        aligned.append(np.column_stack((x, y)))
    return aligned
