import math

import numpy as np
from numba import njit

from preprocess import evenly_spaced_resample, preprocess_centerline
from scaled_parameters import ALIGNMENT_PENALTY


@njit
def get_sliding_windows(y_1, y_2, h_offset, window_len):
    assert len(y_1) >= len(y_2)
    start_1 = max(0, h_offset)
    start_2 = max(0, -h_offset)
    end_1 = min(len(y_1), len(y_2) + h_offset)
    end_2 = min(len(y_2), len(y_1) - h_offset)
    windows_1 = np.lib.stride_tricks.sliding_window_view(y_1[start_1:end_1], window_len)
    windows_2 = np.lib.stride_tricks.sliding_window_view(y_2[start_2:end_2], window_len)
    windows_1 = np.ascontiguousarray(windows_1)
    windows_2 = np.ascontiguousarray(windows_2)
    return windows_1, windows_2


@njit
def numba_mean_last_axis(arr):
    means = []
    for x in arr:
        means.append(np.mean(x))
    return np.array(means)


@njit
def compute_v_offset_range(y_1, y_2, h_offset, window_len):
    windows_1, windows_2 = get_sliding_windows(y_1, y_2, h_offset, window_len)
    mean_1 = numba_mean_last_axis(windows_1)
    mean_2 = numba_mean_last_axis(windows_2)
    v_offsets = mean_1 - mean_2
    min_v_offset = v_offsets.min()
    max_v_offset = v_offsets.max()
    return min_v_offset, max_v_offset


@njit
def numba_quantile_last_axis(arr, q):
    quantiles = []
    for x in arr:
        quantiles.append(np.quantile(x, q))
    return np.array(quantiles)


@njit
def quantile_error(y_1, y_2, h_offset, window_len, quantile):
    windows_1, windows_2 = get_sliding_windows(y_1, y_2, h_offset, window_len)
    diff = windows_1 - windows_2
    errors = np.sum(diff**2, axis=1)
    final_error = np.quantile(errors, quantile)
    return final_error


@njit(parallel=True, fastmath=True)
def compute_quantile_errors(windows_1, windows_2, v_offsets, quantile):
    n_windows, window_len = windows_1.shape
    n_offsets = len(v_offsets)
    diff = windows_1 - windows_2
    cst = np.sum(diff**2, axis=1)
    cst = np.broadcast_to(cst, (n_offsets, n_windows))
    lin = np.outer(v_offsets, 2 * np.sum(diff, axis=1))
    quad = window_len * v_offsets**2
    quad = np.repeat(quad, n_windows).reshape((n_offsets, n_windows))
    errors = cst + lin + quad
    quantile_errors = numba_quantile_last_axis(errors, quantile)
    return quantile_errors


@njit
def align_quantile(y_1, y_2, max_translation, penalty, window, quantile, pixel_size):
    if len(y_2) > len(y_1):
        best_h_offset, best_v_offset = align_quantile(
            y_2, y_1, max_translation, penalty, window, quantile, pixel_size
        )
        return -best_h_offset, -best_v_offset
    y_1 = y_1.astype(np.float32)
    y_2 = y_2.astype(np.float32)
    window_len = round(window * (len(y_2) - 1)) + 1
    physical_length = pixel_size * (window_len - 1)
    min_h_offset = -round(max_translation * (len(y_2) - 1))
    max_h_offset = len(y_2) - len(y_1) + round(max_translation * (len(y_2) - 1))
    best_error = np.inf
    best_h_offset = 0
    best_v_offset = 0
    for h_offset in range(min_h_offset, max_h_offset + 1):
        min_v_offset, max_v_offset = compute_v_offset_range(
            y_1, y_2, h_offset, window_len
        )
        v_offset_sampling = math.ceil(max_v_offset - min_v_offset) + 1
        v_offsets = np.linspace(min_v_offset, max_v_offset, v_offset_sampling)
        stick_out = max(0, -h_offset) + max(0, len(y_1) + h_offset - len(y_2))
        physical_stick_out = stick_out * pixel_size
        stick_out_error = penalty * physical_stick_out

        # windows_1, windows_2 = get_sliding_windows(y_1, y_2, h_offset, window_len)
        # quantile_errors = compute_quantile_errors(
        #     windows_1, windows_2, v_offsets, quantile
        # )
        # i_min = np.argmin(quantile_errors)
        # v_offset = v_offsets[i_min]
        # total_error = quantile_errors[i_min]

        for v_offset in v_offsets:
            error = quantile_error(y_1, y_2 + v_offset, h_offset, window_len, quantile)
            total_error = error / physical_length + stick_out_error
            if total_error < best_error:
                best_error = total_error
                best_h_offset = h_offset
                best_v_offset = v_offset

    return best_h_offset, best_v_offset


def align_with_reference(xs, ys, reference_xs, reference_ys, params, pixel_size):
    params = params.copy()
    max_translation = params.pop("max_translation")
    penalty = params.pop("penalty")
    physical_max_translation = max_translation * pixel_size
    if reference_xs is None or reference_ys is None:
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
    # delta, v_delta = align_min_l2(ref_ys_p, ys_p, max_translation, pixel_size, penalty)
    if len(ref_ys_p) < 5 or len(ys_p) < 5:
        return xs, ys
    delta, v_delta = align_quantile(ref_ys_p, ys_p, 0.33, 0, 0.33, 0.3, pixel_size)
    h_translation = ref_xs_p[0] - xs_p[0] + delta * pixel_size
    return xs + h_translation, ys + v_delta
