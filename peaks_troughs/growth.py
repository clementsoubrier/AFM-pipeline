import os

import matplotlib.pyplot as plt
import numpy as np

from align import align_centerlines
from group_by_cell import Orientation, load_cell
from preprocess import get_scaled_parameters, int_resample, preprocess_centerline


def plot_cell(cell, name):
    cell_centerlines = []
    for frame_data in cell:
        xs = frame_data["xs"]
        ys = frame_data["ys"]
        if frame_data["orientation"] is Orientation.NEW_POLE_OLD_POLE:
            xs = xs[-1] - xs[::-1]
            ys = ys[::-1]
        xs, ys = int_resample(xs, ys)
        pixel_size = frame_data["pixel_size"]
        params = get_scaled_parameters(pixel_size, misc=True)
        max_translation = params.pop("max_translation")
        del params["v_offset"]
        centerline = preprocess_centerline(xs, ys, **params)
        xs, ys = centerline
        cell_centerlines.append(centerline)
        cell_centerlines.append((xs, ys))

    cell_centerlines = align_centerlines(*cell_centerlines,
                                         max_translation=max_translation)
    xi = cell_centerlines[0][0, 0]
    xf = cell_centerlines[0][-1, 0]
    old_pole = []
    new_pole = []
    plt.figure()
    offset = 0
    for centerline in cell_centerlines:
        old_pole.append(xi - centerline[0, 0])
        new_pole.append(centerline[-1, 0] - xf)
        xs = centerline[:, 0]
        ys = centerline[:, 1]
        plt.plot(xs, ys - np.median(ys) + offset)
        offset += 15
    plt.title("  ".join(name))
    plt.figure()
    plt.plot(old_pole)
    plt.plot(new_pole)
    plt.show()



def main():
    cells = []
    datasets = ["05-02-2014", "05-10-2015", "06-10-2015", "30-03-2015"]
    datasets = [os.path.join("WT_mc2_55", dataset) for dataset in datasets]
    all_paths = []
    for dataset in datasets:
        path = os.path.join("data", "datasets", dataset, "Height", "Dic_dir")
        roi_dict = np.load(os.path.join(path, "ROI_dict.npy"),
                        allow_pickle=True).item()
        for roi_name, roi in roi_dict.items():
            if roi["Parent"] and roi["Children"]:
                cell = load_cell(roi_name, dataset)
                if all(frame_data["orientation"] is not Orientation.UNKNOWN
                       for frame_data in cell) and cell:
                    cells.append(cell)
                    all_paths.append((roi_name, dataset))
    for cell, name in zip(cells, all_paths):
        plot_cell(cell, name)


if __name__ == '__main__':
    main()
