from math import dist
import os
from PIL import Image

import numpy as np
import tqdm

from preprocess import preprocess_centerline


def get_centerlines_by_cell(dataset, progress_bar=True):
    path = os.path.join("data", "cells", dataset)
    if progress_bar:
        directories = tqdm.tqdm(sorted(os.listdir(path)))
    else:
        directories = sorted(os.listdir(path))
    for cell_dirname in directories:
        centerlines = os.listdir(os.path.join(path, cell_dirname))
        cell = []
        for filename in centerlines:
            filename = os.path.join(path, cell_dirname, filename)
            with np.load(filename) as centerline:
                xs = centerline["xs"]
                ys = centerline["ys"]
            cell.append((xs, ys))
        yield cell, int(cell_dirname)


def load_data(dataset):
    path = os.path.join("data", "WT_mc2_55", dataset, "Height", "Dic_dir",
                        "Main_dictionnary.npy")
    main_dict = np.load(path, allow_pickle=True).item()
    images = {}
    for img_name, img_dict in main_dict.items():
        img_path = os.path.join("data", img_dict["adress"])
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


def keep_centerline(xs, ys, min_len, kernel_len, std_cut, window, min_prep_len,
                    max_der_std, max_der):
    if xs[-1] - xs[0] < min_len:
        return False
    corrupted = np.logical_or(ys <= 2, ys >= 253)
    if 20 * np.count_nonzero(corrupted) >= len(ys):
        return False
    xs_p, _ = preprocess_centerline(xs, ys, kernel_len, std_cut, window)
    if xs_p[-1] - xs_p[0] < min_prep_len:
        return False
    start = np.searchsorted(xs, xs_p[0], "right") - 1
    end = np.searchsorted(xs, xs_p[-1], "left") + 1
    dx = xs[start + 1: end] - xs[start: end - 1]
    dy = ys[start + 1: end] - ys[start: end - 1]
    der = dy / dx
    mean = np.mean(der)
    std = np.std(der)
    if np.any(abs(der - mean) >= min(max_der_std * std, max_der)):
        return False
    return True


def find_children(mask_id, img_dict):
    children = []
    links = img_dict.get("basic_graph", [])
    for u, v in links:
        if u == mask_id + 1:
            children.append((img_dict["child"], v - 1))
    return children


def save_cell(cell, cell_dirname, main_dict, images, seen, min_len, kernel_len,
              std_cut, window, min_prep_len, max_der_std, max_der):
    os.makedirs(cell_dirname)
    frame_cnt = 0
    children = [cell]
    ends = (0, 0, 0, 0)
    while len(children) == 1:
        img, mask_id = children.pop()
        seen.add((img, mask_id))
        img_dict = main_dict[img]
        centerline = img_dict["centerlines"][mask_id]
        if not centerline.size:
            break
        afm_img = images[img]
        centerline, ends = enforce_direction(centerline, *ends)
        xs, ys = extract_height_centerline(centerline, afm_img)
        if keep_centerline(xs, ys, min_len, kernel_len, std_cut, window,
                           min_prep_len, max_der_std, max_der):
            filename = os.path.join(cell_dirname, f"{frame_cnt:02d}.npz")
            np.savez(filename, xs=xs, ys=ys)
            frame_cnt += 1
        children = find_children(mask_id, img_dict)
    if frame_cnt:
        return 1
    os.rmdir(cell_dirname)
    return 0


def main():
    dataset = "05-02-2014"
    min_len = 40
    kernel_len = 3
    std_cut = 2.5
    window = 3
    min_prep_len = 35
    max_der_std = 5
    max_der = 15
    main_dict, images = load_data(dataset)
    cell_cnt = 0
    seen = set()
    for img, img_dict in tqdm.tqdm(main_dict.items(), total=len(main_dict)):
        for mask_id in range(len(img_dict["centerlines"])):
            if (img, mask_id) in seen:
                continue
            cell_dirname = os.path.join("data", "cells", dataset,
                                        f"{cell_cnt:04d}")
            cell = img, mask_id
            cell_cnt += save_cell(cell, cell_dirname, main_dict, images, seen,
                                  min_len, kernel_len, std_cut, window,
                                  min_prep_len, max_der_std, max_der)


if __name__ == '__main__':
    main()
