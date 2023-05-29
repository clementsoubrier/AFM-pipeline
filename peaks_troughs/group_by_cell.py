from enum import IntEnum
import glob
import math
import os
from PIL import Image
import statistics

import numpy as np
import tqdm

from preprocess import get_scaled_parameters, keep_centerline


class Orientation(IntEnum):
    UNKNOWN = 0
    OLD_POLE_NEW_POLE = 1
    NEW_POLE_OLD_POLE = 2


def get_centerlines_by_cell(dataset, progress_bar=True, return_defects=False):
    path = os.path.join("data", "cells", dataset)
    directories = os.listdir(path)
    if progress_bar:
        directories = tqdm.tqdm(directories, desc=dataset)
    for roi_dirname in directories:
        centerlines = os.listdir(os.path.join(path, roi_dirname))
        roi = []
        for filename in centerlines:
            filename = os.path.join(path, roi_dirname, filename)
            with np.load(filename, allow_pickle=True) as centerline:
                xs = centerline["xs"]
                ys = centerline["ys"]
                timestamp = centerline["timestamp"].item()
                pixel_size = centerline["pixel_size"].item()
                no_defect = centerline["no_defect"].item()
                orientation = centerline["orientation"].item()
                frame_data = {"xs": xs, "ys": ys, "timestamp": timestamp,
                              "pixel_size": pixel_size,
                              "orientation": Orientation(orientation)}
            if no_defect or return_defects:
                roi.append(frame_data)
        roi_id = int(roi_dirname.split("ROI ")[-1])
        if roi:
            yield roi_id, roi


def load_data(dataset):
    path = os.path.join("data", "datasets", dataset, "Height", "Dic_dir")
    main_dict = np.load(os.path.join(path, "Main_dictionnary.npy"),
                        allow_pickle=True).item()
    masks_list = np.load(os.path.join(path, "masks_list.npy"),
                         allow_pickle=True)
    roi_dict = np.load(os.path.join(path, "ROI_dict.npy"),
                       allow_pickle=True).item()
    images = {}
    for img_name, img_dict in main_dict.items():
        img_path = os.path.join("data", "datasets", img_dict["adress"])
        with Image.open(img_path) as img:
            img = np.array(img)
            images[img_name] = img
    return main_dict, masks_list, roi_dict, images


def use_same_direction(reference, line):
    diffs = reference[:, np.newaxis] - line[np.newaxis]
    dists_sq = np.sum(diffs ** 2, axis=-1)
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


def extract_height_centerline(centerline, img):
    xs = [0]
    for i in range(1, len(centerline)):
        x = xs[-1] + math.dist(centerline[i], centerline[i - 1])
        xs.append(x)
    xs = np.array(xs, dtype=np.float64)
    ys = [img[i, j] for i, j in centerline]
    ys = np.array(ys, dtype=np.float64)
    return xs, ys


def save_mask(img_dict, afm_img, mask_id, reference, orientation, roi_dirname):
    line = img_dict["centerlines"][mask_id]
    if not line.size:
        return reference, orientation
    if reference is not None:
        line = use_same_direction(reference, line)
    if orientation is None:
        orientation = determine_orientation(reference, line)
    reference = line
    timestamp = img_dict["time"]
    pixel_size = img_dict["resolution"]
    xs, ys = extract_height_centerline(line, afm_img)
    params = get_scaled_parameters(pixel_size, filtering=True)
    no_defect = int(keep_centerline(xs, ys, **params))
    mask_num = len(os.listdir(roi_dirname))
    filename = f"{mask_num:03d}.npz"
    path = os.path.join(roi_dirname, filename)
    np.savez(path, xs=xs, ys=ys, timestamp=timestamp, pixel_size=pixel_size,
             no_defect=no_defect, orientation=orientation.value)
    return line, orientation


def save_roi(roi, reference, masks_list, main_dict, images, roi_dirname):
    os.makedirs(roi_dirname)
    masks = roi["Mask IDs"]
    if reference is None:
        orientation = Orientation.UNKNOWN
    else:
        orientation = None
    for mask in masks:
        _, _, frame, mask_label = masks_list[mask]
        img_dict = main_dict[frame]
        afm_img = images[frame]
        reference, orientation = save_mask(img_dict, afm_img, mask_label - 1,
                                           reference, orientation, roi_dirname)
    return reference
        

def save_dataset(dataset):
    main_dict, masks_list, roi_dict, images = load_data(dataset)
    references = {}
    for roi_name, roi in tqdm.tqdm(roi_dict.items(), desc=dataset,
                                   total=len(roi_dict)):
        roi_dirname = os.path.join("data", "cells", dataset, roi_name)
        reference = references.pop(roi_name, None)
        reference = save_roi(roi, reference, masks_list, main_dict, images,
                             roi_dirname)
        for child in roi["Children"]:
            references[child] = reference


def main():
    datasets = None
    dataset = None

    if dataset is None:
        if datasets is None:
            datasets_dir = os.path.join("data", "datasets")
            pattern = os.path.join(datasets_dir, "**", "Height", "")
            datasets = glob.glob(pattern, recursive=True)
            datasets = [os.path.dirname(os.path.relpath(path, datasets_dir))
                        for path in datasets]
    else:
        datasets = [dataset]

    for dataset in datasets:
        save_dataset(dataset)


if __name__ == '__main__':
    main()
