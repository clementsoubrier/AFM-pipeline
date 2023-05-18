import itertools
import numbers

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np

from align import align_centerlines
from derivative_sign_segmentation import find_peaks_troughs, PEAK, TROUGH
from group_by_cell import get_centerlines_by_cell
from preprocess import get_scaled_parameters, preprocess_centerline, \
    REF_PIXEL_SIZE, REF_VERTI_SCALE


TEST_PLOT_KYMOGRAPH = -1
PLOT_NONE = 0
PLOT_CENTERLINE = 1
PLOT_CELL = 2
PLOT_3D = 3


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
        xs = centerline[:, 0]
        ys = centerline[:, 1]
        points = []
        for l, r in areas:
            i = np.searchsorted(xs, l, "right") - 1
            j = np.searchsorted(xs, r, "left")
            k = extremum(range(i, j + 1), key=ys.__getitem__)
            x = xs[k]
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
    plt.xlabel("Curvilign abscissa (µm)")
    plt.ylabel("Height")
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
    z_max = max(centerline[:, 1].max() for centerline in cell_centerlines)
    z_min = min(centerline[:, 0].min() for centerline in cell_centerlines)
    z_offset = 0.01 * (z_max - z_min)
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
        ax.plot3D(xs, [i] * size, zs + z_offset, 'black')
    ax.plot_surface(xs_3d, ys_3d, zs_3d, cmap="cividis", lw=0.5, rstride=1,
                    cstride=1, alpha=0.7, edgecolor='none',
                    norm=mplc.PowerNorm(gamma=0.6))
    if cell_peaks is not None:
        for i, peaks in enumerate(cell_peaks):
            if peaks.size:
                ax.scatter(peaks[:, 0], [i] * len(peaks), peaks[:, 1] +
                           z_offset, c="red", edgecolors='none', s=42)
    if cell_troughs is not None:
        for i, troughs in enumerate(cell_troughs):
            if troughs.size:
                ax.scatter(troughs[:, 0], [i] * len(troughs), troughs[:, 1] + 
                           z_offset, c="green", edgecolors='none', s=42)
    if cell_id is not None:
        ax.set_title(str(cell_id))
    plt.show()


def plot_kymograph(*cells, scale=(REF_PIXEL_SIZE, REF_VERTI_SCALE),
                   peaks_and_troughs=True, title=None, smooth=False):
    if isinstance(scale, numbers.Real):
        scales = itertools.repeat((scale, REF_VERTI_SCALE))
    elif isinstance(scale, tuple) and len(scale) == 2 and \
            isinstance(scale[0], numbers.Real) and \
            isinstance(scale[1], numbers.Real):
        scales=itertools.repeat(scale)
    elif isinstance(scale, tuple) and len(scale) == 2 and \
            isinstance(scale[0], numbers.Real) and scale[1] is None:
        scales = itertools.repeat((scale[0], REF_VERTI_SCALE))
    else:
        scales = itertools.chain.from_iterable(scale)
    if smooth:
        all_centerlines = []
        for centerline, scale in zip(itertools.chain.from_iterable(cells),
                                     scales):
            params = get_scaled_parameters(*scale)
            centerline = preprocess_centerline(*centerline, **params)
            all_centerlines.append(centerline)
    else:
        all_centerlines = list(itertools.chain.from_iterable(cells))
    all_centerlines = list(map(np.column_stack, all_centerlines))
    if peaks_and_troughs:
        cell_peaks = []
        cell_troughs = []
        for centerline, (pixel_size, verti_scale) in zip(all_centerlines,
                                                         scales):
            xs = centerline[:, 0]
            ys = centerline[:, 1]
            params = get_scaled_parameters(pixel_size, verti_scale,
                                           peaks_troughs=True)
            _, _, peaks, troughs = find_peaks_troughs(xs, ys, **params,
                                                      smoothing=False)
            if verti_scale is None:
                verti_scale = REF_VERTI_SCALE
            xs *= pixel_size
            ys *= verti_scale
            peaks = [(l * pixel_size, r * pixel_size) for l, r in peaks]
            troughs = [(l * pixel_size, r * pixel_size) for l, r in troughs]
            cell_peaks.append(peaks)
            cell_troughs.append(troughs)
        all_peaks = areas_to_points(cell_peaks, all_centerlines, PEAK)
        all_troughs = areas_to_points(cell_troughs, all_centerlines, TROUGH)
    else:
        all_peaks = None
        all_troughs = None
    plot_3d_centerlines(all_centerlines, all_peaks, all_troughs, title)


def main():
    dataset = "30-03-2015"
    plot_mode = PLOT_CENTERLINE

    if plot_mode == TEST_PLOT_KYMOGRAPH:
        cells = []
        for cell, scales, cell_id in get_centerlines_by_cell(dataset, False):
            if cell_id == 5:
                break
            cell_centerlines = []
            for (xs, ys), (pixel_size, verti_scale) in zip(cell, scales):
                params = get_scaled_parameters(pixel_size, verti_scale,
                                               misc=True)
                max_translation = params.pop("max_translation")
                del params["v_offset"]
                centerline = preprocess_centerline(xs, ys, **params)
                cell_centerlines.append(centerline)
            cell = align_centerlines(*cell_centerlines,
                                     max_translation=max_translation)
            cell = [(line[:, 0], line[:, 1]) for line in cell]
            cells.append(cell)
        plot_kymograph(*cells)
        plot_kymograph(*cells, peaks_and_troughs=False)
        plot_kymograph(*cells, title="smoothing", smooth=True)
        return

    for cell, scales, cell_id in get_centerlines_by_cell(dataset):
        cell_centerlines = []
        for (xs, ys), (pixel_size, verti_scale) in zip(cell, scales):
            params = get_scaled_parameters(pixel_size, verti_scale,
                                           misc=True)
            max_translation = params.pop("max_translation")
            v_offset = params.pop("v_offset")
            centerline = preprocess_centerline(xs, ys, **params)
            cell_centerlines.append(centerline)
        cell_centerlines = align_centerlines(*cell_centerlines,
                                             max_translation=max_translation)
        cell_peaks = []
        cell_troughs = []
        for centerline, (pixel_size, verti_scale) in zip(cell_centerlines,
                                                         scales):
            xs = centerline[:, 0]
            ys = centerline[:, 1]
            params = get_scaled_parameters(pixel_size, verti_scale,
                                           peaks_troughs=True)
            xs, ys, peaks, troughs = find_peaks_troughs(xs, ys, **params,
                                                        smoothing=False)
            if verti_scale is None:
                verti_scale = REF_VERTI_SCALE
            xs *= pixel_size
            ys *= verti_scale
            peaks = [(l * pixel_size, r * pixel_size) for l, r in peaks]
            troughs = [(l * pixel_size, r * pixel_size) for l, r in troughs]
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
            plot_3d_centerlines(cell_centerlines, cell_peaks, cell_troughs,
                                cell_id)


if __name__ == "__main__":
    main()
