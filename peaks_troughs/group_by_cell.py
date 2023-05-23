import glob
from math import dist
import os
from PIL import Image

import numpy as np
import tqdm

from preprocess import get_scaled_parameters, keep_centerline


def get_centerlines_by_cell(dataset, progress_bar=True, return_scale=True):
    path = os.path.join("data", "cells", dataset)
    if progress_bar:
        directories = tqdm.tqdm(sorted(os.listdir(path)), desc=dataset)
    else:
        directories = sorted(os.listdir(path))
    for cell_dirname in directories:
        centerlines = os.listdir(os.path.join(path, cell_dirname))
        cell = []
        scales = []
        for filename in centerlines:
            filename = os.path.join(path, cell_dirname, filename)
            with np.load(filename, allow_pickle=True) as centerline:
                xs = centerline["xs"]
                ys = centerline["ys"]
                scale = centerline["scale"]
            cell.append((xs, ys))
            scales.append(scale)
        if return_scale:
            yield cell, scales, int(cell_dirname)
        else:
            yield cell, int(cell_dirname)


def load_data(dataset):
    path = os.path.join("data", "datasets", dataset, "Height", "Dic_dir",
                        "Main_dictionnary.npy")
    main_dict = np.load(path, allow_pickle=True).item()
    images = {}
    for img_name, img_dict in main_dict.items():
        img_path = os.path.join("data", "datasets", img_dict["adress"])
        with Image.open(img_path) as img:
            images[img_name] = np.array(img)
    return main_dict, images


def enforce_direction(centerline, prev_x_1, prev_y_1, prev_x_2, prev_y_2):
    x_1, y_1 = centerline[0]
    x_2, y_2 = centerline[-1]
    d_11 = (x_1 - prev_x_1) ** 2 + (y_1 - prev_y_1) ** 2
    d_12 = (x_1 - prev_x_2) ** 2 + (y_1 - prev_y_2) ** 2
    d_21 = (x_2 - prev_x_1) ** 2 + (y_2 - prev_y_1) ** 2
    d_22 = (x_2 - prev_x_2) ** 2 + (y_2 - prev_y_2) ** 2
    if d_11 + d_22 > d_12 + d_21:
        prev_x_1, prev_y_1 = x_1, y_1
        prev_x_2, prev_y_2 = x_2, y_2
    else:
        centerline = centerline[:: -1]
        prev_x_1, prev_y_1 = x_2, y_2
        prev_x_2, prev_y_2 = x_1, y_1
    return centerline, (x_1, y_1, x_2, y_2)


def extract_height_centerline(centerline, img):
    xs = [0]
    for i in range(1, len(centerline)):
        x = xs[-1] + dist(centerline[i], centerline[i - 1])
        xs.append(x)
    xs = np.array(xs, dtype=np.float64)
    ys = [img[i, j] for i, j in centerline]
    ys = np.array(ys, dtype=np.float64)
    return xs, ys


def find_children(mask_id, img_dict):
    children = []
    links = img_dict.get("basic_graph", [])
    for u, v in links:
        if u == mask_id + 1:
            children.append((img_dict["child"], v - 1))
    return children


def save_cell(cell, cell_dirname, main_dict, images, seen):
    os.makedirs(cell_dirname)
    frame_cnt = 0
    children = [cell]
    ends = (0, 0, 0, 0)
    while len(children) == 1:
        img, mask_id = children.pop()
        seen.add((img, mask_id))
        img_dict = main_dict[img]
        pixel_size = img_dict["resolution"]
        verti_scale = None
        centerline = img_dict["centerlines"][mask_id]
        if not centerline.size:
            break
        afm_img = images[img]
        centerline, ends = enforce_direction(centerline, *ends)
        xs, ys = extract_height_centerline(centerline, afm_img)
        params = get_scaled_parameters(pixel_size, verti_scale, filtering=True)
        if keep_centerline(xs, ys, **params):
            filename = os.path.join(cell_dirname, f"{frame_cnt:02d}.npz")
            scale = np.array([pixel_size, verti_scale], dtype=object)
            np.savez(filename, xs=xs, ys=ys, scale=scale)
            frame_cnt += 1
        children = find_children(mask_id, img_dict)
    if frame_cnt:
        return 1
    os.rmdir(cell_dirname)
    return 0


def save_dataset(dataset):
    main_dict, images = load_data(dataset)
    cell_cnt = 0
    seen = set()
    for img, img_dict in tqdm.tqdm(main_dict.items(), total=len(main_dict),
                                   desc=dataset):
        for mask_id in range(len(img_dict["centerlines"])):
            if (img, mask_id) in seen:
                continue
            cell_dirname = os.path.join("data", "cells", dataset,
                                        f"{cell_cnt:04d}")
            cell = img, mask_id
            cell_cnt += save_cell(cell, cell_dirname, main_dict, images, seen)


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
