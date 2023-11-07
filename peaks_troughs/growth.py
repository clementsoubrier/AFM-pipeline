import itertools
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from group_by_cell import Orientation, load_cell


def extract_growth(roi):
    t_0 = roi[0]["timestamp"]
    roi = iter(roi)
    while True:
        try:
            frame_data = next(roi)
        except StopIteration:
            return [], [], [], [], []
        orientation = Orientation(frame_data["orientation"])
        xs = frame_data["xs"]
        match orientation:
            case Orientation.UNKNOWN:
                continue
            case Orientation.NEW_POLE_OLD_POLE:
                old_pole_0 = xs[-1]
                new_pole_0 = xs[0]
            case Orientation.OLD_POLE_NEW_POLE:
                old_pole_0 = xs[0]
                new_pole_0 = xs[-1]
        break
    timestamps = [0]
    old_pole_growth = [0]
    new_pole_growth = [0]
    for frame_data in roi:
        timestamp = frame_data["timestamp"]
        orientation = Orientation(frame_data["orientation"])
        xs = frame_data["xs"]
        match orientation:
            case Orientation.UNKNOWN:
                continue
            case Orientation.NEW_POLE_OLD_POLE:
                old_pole = xs[-1]
                new_pole = xs[0]
            case Orientation.OLD_POLE_NEW_POLE:
                old_pole = xs[0]
                new_pole = xs[-1]
        timestamps.append(timestamp - t_0)
        old_pole_growth.append(old_pole_0 - old_pole)
        new_pole_growth.append(new_pole - new_pole_0)
    old_pole_speed = []
    new_pole_speed = []
    for (t_1, t_2), (old_1, old_2), (new_1, new_2) in zip(
        *map(itertools.pairwise, [timestamps, old_pole_growth, new_pole_growth])
    ):
        d_t = t_2 - t_1
        d_old = old_2 - old_1
        d_new = new_2 - new_1
        old_pole_speed.append(d_old / d_t)
        new_pole_speed.append(d_new / d_t)
    return timestamps, old_pole_growth, new_pole_growth, old_pole_speed, new_pole_speed


def plot_single_cell_growth(roi):
    (
        timestamps,
        old_pole_growth,
        new_pole_growth,
        old_pole_speed,
        new_pole_speed,
    ) = extract_growth(roi)
    # plt.figure()
    # plt.plot(timestamps, old_pole_growth, label="old pole")
    # plt.plot(timestamps, new_pole_growth, label="new pole")
    # plt.legend()
    overall = np.array(old_pole_growth) + np.array(new_pole_growth)
    # plt.figure()
    # plt.plot(timestamps[1:], old_pole_speed, label="old pole")
    # plt.plot(timestamps[1:], new_pole_speed, label="new pole")
    # plt.legend()
    plt.plot(timestamps, overall)
    plt.show()


# def main():
#     dataset = os.path.join("WT_mc2_55", "30-03-2015")
#     ages = []
#     old_pole_growth = []
#     new_pole_growth = []
#     old_pole_speed = []
#     new_pole_speed = []
#     for roi_name, roi in get_centerlines_by_cell(dataset):
#         dates, old_growth, new_growth, old_speed, new_speed = extract_growth(roi)
#         ages.extend(dates[1:])
#         old_pole_growth.extend(old_growth[1:])
#         new_pole_growth.extend(new_growth[1:])
#         old_pole_speed.extend(old_speed)
#         new_pole_speed.extend(new_speed)
#         if len(dates) >= 5:
#             plt.figure()
#             offset = 0
#             for frame_data in roi:
#                 xs = frame_data["xs"]
#                 ys = frame_data["ys"]
#                 plt.plot(xs, ys + offset)
#                 offset += 100
#             plt.title(roi_name)
#             plt.figure()
#             plot_single_cell_growth(roi)
#     plt.figure()
#     plt.scatter(ages, old_pole_growth, label="old")
#     plt.scatter(ages, new_pole_growth, label="new")
#     plt.legend()
#     # plt.figure()
#     # plt.scatter(ages, old_pole_speed, label="old")
#     # plt.scatter(ages, new_pole_speed, label="new")
#     # plt.legend()
#     plt.show()


def piecewise_pointwise_linear_regression(times, y):
    assert times[-1] - times[0] >= 75
    best_err = np.inf
    best_a_1 = None
    best_a_2 = None
    best_b = None
    best_t = None
    errs = []
    for i, (t_1, t_2) in enumerate(itertools.pairwise(times), 1):
        if t_2 < 45 or t_1 > times[-1] - 30:
            continue
        t_1 = max(t_1, 45)
        t_2 = min(t_2, times[-1] - 30)
        for t in np.linspace(t_1, t_2, round(t_2 - t_1) + 1):
            mat = np.zeros((len(times), 3))
            mat[:i, 0] = times[:i] - t
            mat[i:, 1] = times[i:] - t
            mat[:, 2] = 1
            coefs = np.linalg.lstsq(mat, y, None)[0]
            a_1, a_2, b = coefs
            err = np.sum((y - np.dot(mat, coefs)) ** 2)
            errs.append(err)
            if err < best_err:
                best_err = err
                best_a_1 = a_1
                best_a_2 = a_2
                best_b = b
                best_t = t
    # plt.plot(errs)
    # plt.show()
    return best_a_1, best_a_2, best_b, best_t


# def main():
#     single_cell = True
#     plot_centerlines = False
#     skip_bad_roi = True
#     bad_rois = {3, 8, 48}
#     dataset = os.path.join("WT_mc2_55", "30-03-2015")
#     timestamps = []
#     growth = []
#     for roi_name, roi in get_centerlines_by_cell(dataset):
#         if skip_bad_roi and roi_name in bad_rois:
#             continue
#         cell_timestamps = []
#         cell_growth = []
#         first_frame = roi[0]
#         t_0 = first_frame["timestamp"]
#         l_0 = first_frame["xs"][-1] - first_frame["xs"][0]
#         for frame_data in roi[1:]:
#             t = frame_data["timestamp"]
#             xs = frame_data["xs"]
#             l = xs[-1] - xs[0]
#             cell_timestamps.append(t - t_0)
#             cell_growth.append(l - l_0)
#         lifespan = cell_timestamps[-1] - cell_timestamps[0]
#         if lifespan < 60:
#             continue
#         times = []
#         lengths = []
#         for frame_data in roi:
#             lengths.append(frame_data["xs"][-1] - frame_data["xs"][0])
#             times.append(frame_data["timestamp"])
#         times = np.array(times) - times[0]
#         lengths = np.array(lengths)
#         a_1, a_2, b, t = piecewise_linear_regression(times, lengths)
#         timestamps.extend(cell_timestamps)
#         growth.extend(cell_growth)
#         if single_cell:
#             plt.figure()
#             # plt.plot(cell_timestamps, cell_growth)
#             plt.plot(times, lengths)
#             plt.plot([0, t, times[-1]], [b - a_1 * t, b, b + a_2 * (times[-1] - t)])
#             plt.title(f"ROI {roi_name} -- ratio {a_2 / a_1:.2f} -- T_0 {t:.2f}")
#             if plot_centerlines:
#                 plt.figure()
#                 offset = 0
#                 for frame_data in roi:
#                     xs = frame_data["xs"]
#                     ys = frame_data["ys"]
#                     plt.plot(xs, ys + offset)
#                     offset += 100
#             plt.show()
#     plt.scatter(timestamps, growth)
#     plt.show()


def surf_growth(ROI, ROI_dict, main_dict, masks_list):
    ID_list = ROI_dict[ROI]["Mask IDs"]
    number = len(ID_list)
    surf_val = np.zeros(number)
    time_val = np.zeros(number)
    resolution = main_dict[masks_list[ID_list[0], 2]]["resolution"]

    for i in tqdm.trange(number):
        elem = ID_list[i]
        frame = masks_list[elem][2]
        time_val[i] = main_dict[frame]["time"]
        surf_val[i] = (
            main_dict[frame]["area"][masks_list[elem][3] - 1] * resolution**2
        )
    return time_val, surf_val


def get_cells_surface(dataset, skip_bad_roi, bad_rois):
    path = os.path.join( "data", "datasets", dataset)
    main_dict_path = os.path.join(path, "Main_dictionnary.npz")
    main_dict = np.load(main_dict_path, allow_pickle=True)["arr_0"].item()
    masks_list_path = os.path.join(path, "masks_list.npz")
    masks_list = np.load(masks_list_path, allow_pickle=True)["arr_0"]
    roi_dict_path = os.path.join(path, "ROI_dict.npz")
    roi_dict = np.load(roi_dict_path, allow_pickle=True)["arr_0"].item()
    individuals = []
    for roi in roi_dict:
        if skip_bad_roi and int(roi.split()[-1]) in bad_rois:
            continue
        timestamps, y = surf_growth(roi, roi_dict, main_dict, masks_list)
        # if y[-1] >= 700:
        #     continue
        y -= y[0]
        lifespan = timestamps[-1] - timestamps[0]
        if lifespan >= 60:
            individuals.append((timestamps, y))
    return individuals


def get_cells_length(dataset, skip_bad_roi, bad_rois, inh_700=False):
    path = os.path.join( "data", "datasets", dataset)
    roi_dict_path = os.path.join(path, "ROI_dict.npz")
    roi_dict = np.load(roi_dict_path, allow_pickle=True)["arr_0"].item()
    full_life_rois = []
    for roi_name, roi in roi_dict.items():
        if roi["Parent"]:  # and roi["Children"]:
            full_life_rois.append(int(roi_name.split()[-1]))
    individuals = []
    for roi_name in full_life_rois:
        roi = load_cell(roi_name, dataset, return_defects=True)
        if len(roi) < 4 or (skip_bad_roi and roi_name in bad_rois):
            continue
        if inh_700 and roi[-1]["timestamp"] >= 700:
            continue
        t_0 = roi[0]["timestamp"]
        timestamps = []
        y = []
        pole_1 = []
        pole_2 = []
        for frame_data in roi:
            timestamps.append(frame_data["timestamp"] - t_0)
            xs = frame_data["xs"]
            length = xs[-1] - xs[0]
            y.append(length)
            pole_1.append(-xs[0])
            pole_2.append(xs[-1])
        match roi[0]["orientation"]:
            case Orientation.UNKNOWN:
                raise RuntimeError
            case Orientation.NEW_POLE_OLD_POLE:
                pole_1, pole_2 = pole_2, pole_1
        lifespan = timestamps[-1] - timestamps[0]
        if lifespan >= 150:
            individuals.append((timestamps, y))
            # individuals.append((timestamps, pole_1))
            # individuals.append((timestamps, pole_2))
            plt.figure()
            plt.plot(timestamps, (np.array(y) - y[0]) / 2 + 1, label="cell")
            plt.plot(timestamps, np.array(pole_1) - pole_1[0], label="old pole")
            plt.plot(timestamps, np.array(pole_2) - pole_2[0], label="new pole")
            plt.legend()
            # plt.figure()
            # offset = 0
            # for frame_data in roi:
            #     plt.plot(frame_data["xs"], frame_data["ys"] + offset)
            #     offset += 50
            plt.show()
    return individuals


def remove_worst_length_outliers(times, y):
    dy = y[1:] - y[:-1]
    dt = times[1:] - times[:-1]
    dydt = dy / dt
    d2ydt = dydt[1:] - dydt[:-1]
    dtbis = (times[2:] - times[:-2]) / 2
    d2ydt2 = d2ydt / dtbis
    q1, q3 = np.percentile(d2ydt2, [25, 75])
    clipped = np.clip(d2ydt2, q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    errors = np.abs(d2ydt2 - clipped)
    i = np.argmax(errors)
    if errors[i] > 0:
        times = np.concatenate([times[: i + 1], times[i + 2 :]])
        y = np.concatenate([y[: i + 1], y[i + 2 :]])
    return times, y


def remove_length_outliers(times, y):
    while len(times) > 0:
        new_times, new_y = remove_worst_length_outliers(times, y)
        if len(new_times) == len(times):
            return times, y
        times = new_times
        y = new_y
    raise RuntimeError


def measure_growth(individuals, show_single_cell=False, skip_bad_roi=True):
    all_a = []
    all_a_1 = []
    all_a_2 = []
    all_b = []
    all_t = []
    ratios = []
    for times, y in individuals:
        y = np.array(y)
        times = np.array(times)
        times, y = remove_length_outliers(times, y)
        a_1, a_2, b, t = piecewise_pointwise_linear_regression(times, y)
        a = statistics.linear_regression(times, y).slope
        # if skip_bad_roi and (a_1 < 0 or a_2 < a_1):
        #     continue
        if a < 0 or a_2 < a_1:
            continue
        t_0 = times[0]
        t_f = times[-1]
        all_a.append(a)
        all_a_1.append(a_1)
        all_a_2.append(a_2)
        all_b.append(b)
        all_t.append(t)
        ratios.append(a_2 / a_1)
        if show_single_cell:
            # dy = y[1:] - y[:-1]
            # dt = times[1:] - times[:-1]
            # speed = dy / dt
            #
            # mid_times = (times[1:] + times[:-1]) / 2
            # dspeed = speed[1:] - speed[:-1]
            # dt2 = mid_times[1:] - mid_times[:-1]
            # acceleration = dspeed / dt2
            # q1, q3 = np.percentile(acceleration, [25, 75])
            # mark = []
            # for i, acc in enumerate(acceleration, 1):
            #     if acc <= q1 - 1.5 * (q3 - q1) or acc >= q3 + 1.5 * (q3 - q1):
            #         mark.append(i)
            #
            # d2, d4, d5, d8 = np.percentile(speed, [20, 40, 50, 80])
            # sus = []
            # for i, s in enumerate(speed):
            #     if s <= d2 - 1 * (d4 - d2) or s >= d8 + 1.5 * (d8 - d5):
            #         sus.append(i)

            piecewise_times = [t_0, t, t_f]
            p_0 = b - a_1 * t
            p_f = b + a_2 * (t_f - t)
            piecewise_y = [p_0, b, p_f]
            y_min = min(min(y), min(piecewise_y))
            y_max = max(max(y), max(piecewise_y))
            plt.scatter(times, np.array(y), label="cell length", color="C2")
            plt.plot(
                piecewise_times, piecewise_y, label="piecewise linear fit", color="C0"
            )
            plt.plot(
                [t, t],
                [y_min, y_max],
                linestyle="dashed",
                label=f"NETO: {int(t)} minutes",
                color="C1",
            )
            plt.plot([], [], " ", label=f"Slope ratio: {a_2 / a_1:.2f}")
            plt.annotate(
                f"{1000 * a_1:.1f} nm / minute", (t_0 + (0.05 * (t_f - t_0)), p_0)
            )
            plt.annotate(
                f"{1000 * a_2:.1f} nm / minute",
                (t_f - (0.05 * (t_f - t_0)), p_f),
                ha="right",
                va="top",
            )
            plt.legend()
            plt.xlabel("Time since cell birth (minutes)")
            plt.ylabel("Cell length (µm)")
            plt.title("Cell length variation throughout lifespan")
            # for i in sus:
            #     plt.plot(times[i : i + 2], y[i : i + 2], color="black")
            # for i in mark:
            #     plt.scatter([times[i]], [y[i]], color="red")
            # plt.figure()
            # plt.boxplot(acceleration)
            plt.show()
    # _, ((ax_11, ax_12), (ax_21, ax_22)) = plt.subplots(2, 2)
    # ax_11.scatter(all_a_1, [0] * len(all_a_1))
    # ax_12.scatter(all_a_2, [0] * len(all_a_2))
    # ax_21.scatter(all_t, [0] * len(all_t))
    # ax_22.scatter(ratios, [0] * len(ratios))
    # ax_11.set_title("First slope (µm/min)")
    # ax_12.set_title("Second slope (µm/min)")
    # ax_21.set_title("NETO (minutes)")
    # ax_22.set_title("Ratio second slope over first slope")
    # plt.show()
    return all_a, all_a_1, all_a_2, all_t


# def measure_growth(individuals, show_single_cell=False, skip_bad_roi=True):
#     all_a_1 = []
#     all_a_2 = []
#     all_b = []
#     all_t = []
#     ratios = []
#     while individuals:
#         for line_type in ["old pole", "new pole", "full cell"]:
#             times, y = individuals.pop()
#             a_1, a_2, b, t = piecewise_pointwise_linear_regression(times, y)
#             # if skip_bad_roi and (a_1 < 0 or a_2 < a_1):
#             #     continue
#             t_f = times[-1]
#             all_a_1.append(a_1)
#             all_a_2.append(a_2)
#             all_b.append(b)
#             all_t.append(t)
#             ratios.append(a_2 / a_1)
#             if show_single_cell:
#                 plt.plot(
#                     times, np.array(y) - y[0], label=line_type
#                 )  # label="cell length")
#                 # plt.plot(
#                 #     [0, t, t_f],
#                 #     [b - a_1 * t, b, b + a_2 * (t_f - t)],
#                 #     label="piecewise linear fit",
#                 # )
#                 # plt.legend()
#                 # plt.title(f"ratio: {a_2 / a_1:.2f} -- T_0: {t:.1f} min")
#         if a_1 < 0 or a_2 < a_1:
#             plt.clf()
#             continue
#         plt.legend()
#         plt.show()
#     _, ((ax_11, ax_12), (ax_21, ax_22)) = plt.subplots(2, 2)
#     ax_11.scatter(all_a_1, [0] * len(all_a_1))
#     ax_12.scatter(all_a_2, [0] * len(all_a_2))
#     ax_21.scatter(all_t, [0] * len(all_t))
#     ax_22.scatter(ratios, [0] * len(ratios))
#     ax_11.set_title("First slope (µm/min)")
#     ax_12.set_title("Second slope (µm/min)")
#     ax_21.set_title("NETO (minutes)")
#     ax_22.set_title("Ratio second slope over first slope")
#     plt.show()


def main():
    show_single_cell = False
    skip_bad_roi = True
    bad_rois = {3, 8}

    a_plt = []
    a_1_plt = []
    a_2_plt = []
    t_plt = []
    # dataset = "WT_INH_700min_2014"
    dataset = os.path.join("WT_mc2_55", "30-03-2015")
    individuals = get_cells_length(
        dataset, skip_bad_roi=skip_bad_roi, bad_rois=bad_rois
    )
    all_a, all_a_1, all_a_2, all_t = measure_growth(
        individuals, show_single_cell, skip_bad_roi=skip_bad_roi
    )
    a_plt.extend(all_a)
    a_1_plt.extend(all_a_1)
    a_2_plt.extend(all_a_2)
    t_plt.extend(all_t)

    dataset = os.path.join("WT_mc2_55", "06-10-2015")
    individuals = get_cells_length(
        dataset, skip_bad_roi=skip_bad_roi, bad_rois=bad_rois
    )
    all_a, all_a_1, all_a_2, all_t = measure_growth(
        individuals, show_single_cell, skip_bad_roi=skip_bad_roi
    )
    a_plt.extend(all_a)
    a_1_plt.extend(all_a_1)
    a_2_plt.extend(all_a_2)
    t_plt.extend(all_t)

    dataset = os.path.join("WT_mc2_55", "30-03-2015")
    individuals = get_cells_length(
        dataset, skip_bad_roi=skip_bad_roi, bad_rois=bad_rois
    )
    all_a, all_a_1, all_a_2, all_t = measure_growth(
        individuals, show_single_cell, skip_bad_roi=skip_bad_roi
    )
    a_plt.extend(all_a)
    a_1_plt.extend(all_a_1)
    a_2_plt.extend(all_a_2)
    t_plt.extend(all_t)

    dataset = os.path.join("WT_INH_700min_2014")
    individuals = get_cells_length(
        dataset, skip_bad_roi=skip_bad_roi, bad_rois=bad_rois, inh_700=True
    )
    all_a, all_a_1, all_a_2, all_t = measure_growth(
        individuals, show_single_cell, skip_bad_roi=skip_bad_roi
    )
    a_plt.extend(all_a)
    a_plt = 1000 * np.array(a_plt)
    a_mean = round(np.mean(a_plt))
    a_std = round(np.std(a_plt))
    a_1_plt.extend(all_a_1)
    a_1_plt = 1000 * np.array(a_1_plt)
    a_1_mean = round(np.mean(a_1_plt))
    a_1_std = round(np.std(a_1_plt))
    a_2_plt.extend(all_a_2)
    a_2_plt = 1000 * np.array(a_2_plt)
    a_2_mean = round(np.mean(a_2_plt))
    a_2_std = round(np.std(a_2_plt))
    t_plt.extend(all_t)
    neto_mean = round(np.mean(t_plt))
    neto_std = round(np.std(t_plt))


    _, ax = plt.subplots()
    ax.boxplot(a_plt, vert=False, showfliers=False)
    ax.set_xlabel("Cell elongation speed (nm / minute)")
    ax.set_yticklabels([""])
    ax.set_title(f"Overall cell elongation speed")
    _, ax = plt.subplots()
    ax.boxplot([a_1_plt, a_2_plt], showfliers=False)
    ax.set_ylabel("Cell elongation speed (nm / minute)")
    ax.set_xticklabels(["Before NETO", "After NETO"])
    ax.set_title("Comparison of the cell elongation speed before and after NETO")
    _, ax = plt.subplots()
    ax.hist(t_plt, bins=5)
    ax.set_xlabel("Time since cell birth (minutes)")
    ax.set_title("NETO")

    plt.show()


if __name__ == "__main__":
    main()
