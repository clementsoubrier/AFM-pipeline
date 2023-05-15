from collections import Counter

import matplotlib.pyplot as plt

from derivative_sign_segmentation import find_peaks_troughs
from group_by_cell import get_centerlines_by_cell


def compute_stats(dataset, kernel_len, std_cut, window, min_depth):
    peak_counter = Counter()
    trough_counter = Counter()
    peak_lengths = []
    trough_lengths = []
    for cell, _ in get_centerlines_by_cell(dataset):
        for xs, ys in cell:
            _, _, peaks, troughs = find_peaks_troughs(xs, ys, kernel_len,
                                                      std_cut, window,
                                                      min_depth)
            peak_counter[len(peaks)] += 1
            trough_counter[len(troughs)] += 1
            peak_lengths.extend(r - l for l, r in peaks)
            trough_lengths.extend(r - l for l, r in troughs)
    return peak_counter, trough_counter, peak_lengths, trough_lengths


def _plot_counts_histogram(ax, counter, feature, dataset):
    n_centerlines = sum(counter.values())
    mini = min(counter)
    maxi = max(counter)
    percentages = [100 * counter[x] / n_centerlines
                   for x in range(mini, maxi + 1)]
    edges = [x - 0.5 for x in range(mini, maxi + 2)]
    xlabel = f"Number of {feature}"
    ylabel = "Proportion of centerlines (%)"
    title = "Repartition of the number of {} ({} | {} centerlines)".format(
        feature, dataset, n_centerlines)
    ax.stairs(percentages, edges, fill=True)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)


def plot_peak_trough_counters(dataset, peak_counter, trough_counter):
    _, (ax1, ax2) = plt.subplots(1, 2)
    _plot_counts_histogram(ax1, peak_counter, "peaks", dataset)
    _plot_counts_histogram(ax2, trough_counter, "troughs", dataset)


def _plot_lengths_histogram(ax, lengths, feature, dataset):
    xlabel = f"Length of the {feature}s"
    title = f"Probability density of the {feature} length ({dataset} | " \
            f"{len(lengths)} {feature}s)"
    ax.hist(lengths, 50, density=True)
    ax.set(xlabel=xlabel, title=title)


def plot_peak_trough_lengths(dataset, peak_lengths, trough_lengths):
    _, (ax1, ax2) = plt.subplots(2, 1)
    _plot_lengths_histogram(ax1, peak_lengths, "peak", dataset)
    _plot_lengths_histogram(ax2, trough_lengths, "trough", dataset)


def plot_stats(dataset, peak_counter, trough_counter, peak_lengths,
               trough_lengths):
    plot_peak_trough_counters(dataset, peak_counter, trough_counter)
    plot_peak_trough_lengths(dataset, peak_lengths, trough_lengths)
    plt.show()


def main():
    kernel_len = 3
    std_cut = 2.5
    window = 3
    min_depth = 1.5
    dataset = "05-02-2014"
    
    stats = compute_stats(dataset, kernel_len, std_cut, window, min_depth)
    plot_stats(dataset, *stats)

if __name__ == "__main__":
    main()
