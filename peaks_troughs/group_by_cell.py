import glob
import math
import operator
import os
import statistics
from enum import IntEnum

import numpy as np
import tqdm

from peaks_troughs.align import align_with_reference
from peaks_troughs.derivative_sign_segmentation import find_peaks_troughs
from peaks_troughs.preprocess import keep_centerline
from peaks_troughs.scaled_parameters import get_scaled_parameters


class Orientation(IntEnum):
    UNKNOWN = 0
    OLD_POLE_NEW_POLE = 1
    NEW_POLE_OLD_POLE = 2


def load_cell(cell_name, dataset=None, return_defects=False):
    match (cell_name, dataset):
        case (str() | bytes() | os.PathLike(), None):
            dataset = ""
        case (int(), str() | bytes() | os.PathLike()):
            cell_name = f"ROI {cell_name}"
        case (str() | bytes() | os.PathLike(), str() | bytes() | os.PathLike()):
            pass
        case _:
            raise ValueError(
                f"The combination ({cell_name}, {dataset}) does not describe a ROI."
            )
    roi_dir = os.path.join("..", "data", "cells", dataset, cell_name)
    centerlines = os.listdir(roi_dir)

    roi = []
    for filename in centerlines:
        filename = os.path.join(roi_dir, filename)
        with np.load(filename) as centerline:
            line = centerline["line"]
            xs = centerline["xs"]
            ys = centerline["ys"]
            timestamp = centerline["timestamp"].item()
            orientation = centerline["orientation"].item()
            no_defect = centerline["no_defect"].item()
            frame_data = {
                "line": line,
                "xs": xs,
                "ys": ys,
                "timestamp": timestamp,
                "orientation": Orientation(orientation),
                "no_defect": no_defect,
            }
        if no_defect or return_defects:
            roi.append(frame_data)
    roi.sort(key=operator.itemgetter("timestamp"))
    return roi


def get_centerlines_by_cell(dataset, progress_bar=True, return_defects=False):
    path = os.path.join("..", "data", "cells", dataset)
    directories = os.listdir(path)
    if progress_bar:
        directories = tqdm.tqdm(directories, desc=dataset)
    for roi_dir in directories:
        roi = load_cell(roi_dir, dataset, return_defects)
        roi_id = int(roi_dir.split("ROI ")[-1])
        if roi:
            yield roi_id, roi


def load_data(dataset, log_progress):
    path = os.path.join("..", "data", "datasets", dataset)
    if log_progress:
        print("Loading main dictionary.", end="")
    main_dict_path = os.path.join(path, "Main_dictionnary.npz")
    main_dict = np.load(main_dict_path, allow_pickle=True)["arr_0"].item()
    if log_progress:
        print(" Done.")
        print("Loading masks list.", end="")
    masks_list_path = os.path.join(path, "masks_list.npz")
    masks_list = np.load(masks_list_path, allow_pickle=True)["arr_0"]
    if log_progress:
        print(" Done.")
        print("Loading ROI dictionary.", end="")
    roi_dict_path = os.path.join(path, "ROI_dict.npz")
    roi_dict = np.load(roi_dict_path, allow_pickle=True)["arr_0"].item()
    if log_progress:
        print(" Done.")
    frame_dicts = main_dict.items()
    if log_progress:
        frame_dicts = tqdm.tqdm(
            frame_dicts, total=len(main_dict), desc="Loading images"
        )
    images = {}
    for img_name, img_dict in frame_dicts:
        img_path = os.path.join("..", "data", "datasets", img_dict["adress"])
        contents = np.load(img_path)
        fwd_img = contents["Height_fwd"]
        try:
            bwd_img = contents["Height_bwd"]
        except KeyError:
            bwd_img = None
        images[img_name] = (fwd_img, bwd_img)
    return main_dict, masks_list, roi_dict, images


def topological_sort(roi_dict):
    rois = set(roi_dict.keys())
    order = []
    while rois:
        roi = rois.pop()
        while parent := roi_dict[roi]["Parent"]:
            roi = parent
        stack = [roi]
        while stack:
            roi = stack.pop()
            rois.discard(roi)
            order.append(roi)
            children = roi_dict[roi]["Children"]
            stack.extend(children)
    return order


def use_same_direction(reference, line):
    diffs = reference[:, np.newaxis] - line[np.newaxis]
    dists_sq = np.sum(diffs**2, axis=-1)
    if len(line) <= len(reference):
        order = dists_sq.argmin(axis=0)
    else:
        order = dists_sq.argmin(axis=1)
    if statistics.linear_regression(range(len(order)), order).slope < 0:
        line = line[::-1]
    return line


def determine_orientation(reference, line):
    if reference is None:
        return Orientation.UNKNOWN
    dist_start_start = math.dist(reference[0], line[0])
    dist_end_end = math.dist(reference[-1], line[-1])
    if dist_start_start < dist_end_end:
        return Orientation.OLD_POLE_NEW_POLE
    if dist_start_start > dist_end_end:
        return Orientation.NEW_POLE_OLD_POLE
    return Orientation.UNKNOWN


def extract_height_profile(centerline, img, pixel_size):
    xs = [0]
    for i in range(1, len(centerline)):
        x = xs[-1] + math.dist(centerline[i], centerline[i - 1])
        xs.append(x)
    xs = pixel_size * np.array(xs, dtype=np.float64)
    ys = [img[i, j] for i, j in centerline]
    ys = np.array(ys, dtype=np.float64)
    return xs, ys


def save_mask(
    img_dict,
    fwd_img,
    bwd_img,
    mask_id,
    reference_line,
    reference_xs,
    reference_ys,
    orientation,
    roi_dirname,
):
    line = img_dict["centerlines"][mask_id]
    if not line.size:
        return reference_line, reference_xs, reference_ys, orientation
    if reference_line is not None:
        line = use_same_direction(reference_line, line)
    if orientation is None:
        orientation = determine_orientation(reference_line, line)
    timestamp = img_dict["time"]
    pixel_size = img_dict["resolution"]
    params = get_scaled_parameters(pixel_size, filtering=True)
    xs, ys = extract_height_profile(line, fwd_img, pixel_size)
    no_defect = keep_centerline(xs, ys, pixel_size, **params)
    if not no_defect and bwd_img is not None:
        xs_bwd, ys_bwd = extract_height_profile(line, bwd_img, pixel_size)
        no_defect_bwd = keep_centerline(xs_bwd, ys_bwd, pixel_size, **params)
        if no_defect_bwd:
            xs = xs_bwd
            ys = ys_bwd
            no_defect = no_defect_bwd
    params = get_scaled_parameters(pixel_size, aligning=True)
    xs, ys = align_with_reference(
        xs, ys, reference_xs, reference_ys, params, pixel_size
    )
    params = get_scaled_parameters(pixel_size, peaks_troughs=True)
    _, _, peaks, troughs = find_peaks_troughs(xs, ys, **params, resolution=pixel_size)
    mask_num = len(os.listdir(roi_dirname))
    filename = f"{mask_num:03d}.npz"
    path = os.path.join(roi_dirname, filename)
    np.savez(
        path,
        line=line,
        xs=xs,
        ys=ys,
        timestamp=timestamp,
        no_defect=no_defect,
        orientation=orientation.value,
        peaks=peaks,
        troughs=troughs,
    )
    length_current = xs[-1] = xs[0]
    if reference_xs is None:
        length_reference = None
    else:
        length_reference = reference_xs[-1] - reference_xs[0]
    if no_defect or (
        len(line) >= 10
        and (reference_line is None or length_current >= length_reference)
    ):
        return line, xs, ys, orientation
    return reference_line, reference_xs, reference_ys, orientation


def save_roi(
    roi,
    reference_line,
    reference_xs,
    reference_ys,
    masks_list,
    main_dict,
    images,
    roi_dirname,
):
    os.makedirs(roi_dirname)
    masks = roi["Mask IDs"]
    if reference_line is None:
        orientation = Orientation.UNKNOWN
    else:
        orientation = None
    for mask in masks:
        _, _, frame, mask_label = masks_list[mask]
        img_dict = main_dict[frame]
        fwd_img, bwd_img = images[frame]
        reference_line, reference_xs, reference_ys, orientation = save_mask(
            img_dict,
            fwd_img,
            bwd_img,
            mask_label - 1,
            reference_line,
            reference_xs,
            reference_ys,
            orientation,
            roi_dirname,
        )
    return reference_line, reference_xs, reference_ys


def save_dataset(dataset, log_progress):
    if log_progress:
        print("Processing dataset", dataset)
    main_dict, masks_list, roi_dict, images = load_data(dataset, log_progress)
    roi_names = topological_sort(roi_dict)
    if log_progress:
        roi_names = tqdm.tqdm(roi_names, desc="Processing ROIs")
    references = {}
    for roi_name in roi_names:
        roi = roi_dict[roi_name]
        roi_dir = os.path.join("..", "data", "cells", dataset, roi_name)
        reference = references.pop(roi_name, (None, None, None))
        reference = save_roi(roi, *reference, masks_list, main_dict, images, roi_dir)
        for child in roi["Children"]:
            references[child] = reference
    if log_progress:
        print("\n\n")


def main():
    log_progress = True
    datasets = None
    dataset = None
    if dataset is None:
        if datasets is None:
            datasets_dir = os.path.join("..", "data", "datasets")
            pattern = os.path.join(datasets_dir, "**", "final_data", "")
            datasets = glob.glob(pattern, recursive=True)
            datasets = [
                os.path.dirname(os.path.relpath(path, datasets_dir))
                for path in datasets
            ]
    else:
        datasets = [dataset]
    for dataset in datasets:
        save_dataset(dataset, log_progress)


if __name__ == "__main__":
    main()
