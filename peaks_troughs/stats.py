import glob
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from peaks_troughs.derivative_sign_segmentation import find_peaks_troughs
from peaks_troughs.group_by_cell import load_dataset
from scaled_parameters import get_scaled_parameters


def compute_stats(dataset):
    peak_counter = Counter()
    trough_counter = Counter()
    peak_lengths = []
    trough_lengths = []
    maindict=np.load('data/datasets/'+dataset+'/Main_dictionnary.npz', allow_pickle=True)['arr_0'].item()
    pixel_size = maindict[list(maindict.keys())[0]]['resolution']
    for _, cell in load_dataset(dataset):
        for frame_data in cell:
            xs = frame_data["xs"]
            ys = frame_data["ys"]
            
            peaks= frame_data['peaks']
            troughs =frame_data['troughs']
            peak_counter[len(peaks)] += 1
            trough_counter[len(troughs)] += 1
            peak_lengths.extend((r - l) * pixel_size for l, r in peaks)
            trough_lengths.extend((r - l) * pixel_size for l, r in troughs)
    stats = {
        "peak_counter": peak_counter,
        "trough_counter": trough_counter,
        "peak_lengths": peak_lengths,
        "trough_lengths": trough_lengths,
    }
    return stats


def _plot_counts_histogram(ax, counter, feature, dataset):
    n_centerlines = sum(counter.values())
    mini = min(counter)
    maxi = max(counter)
    percentages = [100 * counter[x] / n_centerlines for x in range(mini, maxi + 1)]
    edges = [x - 0.5 for x in range(mini, maxi + 2)]
    xlabel = f"Number of {feature}"
    ylabel = "Proportion of centerlines (%)"
    title = "Repartition of the number of {} ({} | {} centerlines)".format(
        feature, dataset, n_centerlines
    )
    ax.stairs(percentages, edges, fill=True)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)


def plot_peak_trough_counters(dataset, peak_counter, trough_counter):
    _, (ax1, ax2) = plt.subplots(1, 2)
    _plot_counts_histogram(ax1, peak_counter, "peaks", dataset)
    _plot_counts_histogram(ax2, trough_counter, "troughs", dataset)


def _plot_lengths_histogram(ax, lengths, feature, dataset):
    mean = np.mean(lengths)
    q1, med, q3 = np.percentile(lengths, [25, 50, 75])
    xlabel = f"Length of the {feature}s (µm)"
    title = (
        f"Probability density of the {feature} length ({dataset} | "
        f"{len(lengths)} {feature}s)"
    )
    ax.hist(lengths, 40, density=True, color="grey")
    ax.axvline(mean, color="red", label="mean")
    ax.axvline(med, color="blue", label="median")
    ax.axvline(q1, color="green", label="quantiles")
    ax.axvline(q3, color="green")
    ax.legend()
    ax.set(xlabel=xlabel, title=title)


def plot_peak_trough_lengths(dataset, peak_lengths, trough_lengths):
    _, (ax1, ax2) = plt.subplots(2, 1)
    _plot_lengths_histogram(ax1, peak_lengths, "peak", dataset)
    _plot_lengths_histogram(ax2, trough_lengths, "trough", dataset)


def plot_stats(dataset, /, peak_counter, trough_counter, peak_lengths, trough_lengths):
    plot_peak_trough_counters(dataset, peak_counter, trough_counter)
    plot_peak_trough_lengths(dataset, peak_lengths, trough_lengths)
    plt.show()


def main():
    datasets = None
    dataset = None

    if dataset is None:
        if datasets is None:
            cells_dir = os.path.join("data", "cells")
            pattern = os.path.join(cells_dir, "**", "ROI *", "")
            datasets = glob.glob(pattern, recursive=True)
            datasets = {
                os.path.dirname(os.path.relpath(path, cells_dir)) for path in datasets
            }
    else:
        datasets = [dataset]

    for dataset in datasets:
        stats = compute_stats(dataset)
        plot_stats(dataset, **stats)


if __name__ == "__main__":
    main()
