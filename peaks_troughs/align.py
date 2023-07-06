import numpy as np

from peaks_troughs.preprocess import evenly_spaced_resample, preprocess_centerline
from peaks_troughs.scaled_parameters import ALIGNMENT_PENALTY


def align_max_covariance(prev, ys, max_translation):
    if len(ys) > len(prev):
        return -align_max_covariance(ys, prev, max_translation)
    if max_translation is None:
        max_translation = len(ys) // 3
    else:
        max_translation = min(max_translation, len(ys) // 3)
    best_cov = -np.inf
    best_delta = None
    best_v_delta = None
    min_delta = -max_translation
    max_delta = len(prev) + max_translation - len(ys)
    for delta in range(min_delta, max_delta + 1):
        a = prev[max(0, delta) : min(len(prev), delta + len(ys))]
        b = ys[max(0, -delta) : len(ys) - max(0, delta + len(ys) - len(prev))]
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        v_delta = mean_a - mean_b
        cov = np.dot(a - mean_a, b - mean_b)
        if cov > best_cov:
            best_cov = cov
            best_delta = delta
            best_v_delta = v_delta
    return best_delta, best_v_delta


def align_min_l2(prev, ys, max_translation, resolution, penalty):
    if len(ys) > len(prev):
        best_delta, best_v_delta = align_min_l2(
            ys, prev, max_translation, resolution, penalty
        )
        return -best_delta, -best_v_delta
    if max_translation is None:
        max_translation = len(ys) // 3
    else:
        max_translation = min(max_translation, len(ys) // 3)
    best_err = np.inf
    best_delta = None
    best_v_delta = None
    min_delta = -max_translation
    max_delta = len(prev) + max_translation - len(ys)
    length_prev = len(prev) * resolution
    length_ys = len(ys) * resolution
    for delta in range(min_delta, max_delta + 1):
        physical_delta = delta * resolution
        a = prev[max(0, delta) : min(len(prev), delta + len(ys))]
        b = ys[max(0, -delta) : len(ys) - max(0, delta + len(ys) - len(prev))]
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        v_delta = mean_a - mean_b
        diff = (a - mean_a) - (b - mean_b)
        stick_out = max(0, -physical_delta) + max(
            0, length_ys + physical_delta - length_prev
        )
        dist_err = np.sqrt(np.dot(diff, diff) / length_ys)
        stick_out_err = penalty * stick_out / np.sqrt(length_ys)
        err = dist_err + stick_out_err
        if err < best_err:
            best_err = err
            best_delta = delta
            best_v_delta = v_delta
    return best_delta, best_v_delta


def align_centerlines(
    resolution, *centerlines, max_translation=None, penalty=ALIGNMENT_PENALTY
):
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
            max_translation_ = min(len(prev), len(y)) // 3
        else:
            max_translation_ = max_translation
        delta, v_delta = align_min_l2(prev, y, resolution, max_translation_, penalty)
        h_translation = delta * resolution + prev_x_0
        x += h_translation
        y += v_delta
        prev_x_0 = x[0]
        prev = y
        centerline = np.column_stack((x, y))
        aligned.append(centerline)
    return aligned


def align_with_reference(xs, ys, reference_xs, reference_ys, params, pixel_size):
    params = params.copy()
    max_translation = params.pop("max_translation")
    penalty = params.pop("penalty")
    physical_max_translation = max_translation * pixel_size
    if (
        reference_xs is None
        or reference_ys is None
        or xs[-1] - xs[0] < physical_max_translation
    ):
        return xs, ys
    xs_p, ys_p = preprocess_centerline(xs, ys, **params, resolution=pixel_size)
    ref_xs_p, ref_ys_p = preprocess_centerline(
        reference_xs, reference_ys, **params, resolution=pixel_size
    )
    if xs_p[-1] - xs_p[0] < physical_max_translation:
        xs_p, ys_p = evenly_spaced_resample(xs, ys, pixel_size)
    if ref_xs_p[-1] - ref_xs_p[0] < physical_max_translation:
        ref_xs_p, ref_ys_p = evenly_spaced_resample(
            reference_xs, reference_ys, pixel_size
        )
    delta, v_delta = align_min_l2(ref_ys_p, ys_p, max_translation, pixel_size, penalty)
    h_translation = ref_xs_p[0] - xs_p[0] + delta * pixel_size
    return xs + h_translation, ys + v_delta
