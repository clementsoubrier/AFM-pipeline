import glob
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import splev, splrep


package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from peaks_troughs.group_by_cell import load_dataset, get_peak_troughs_lineage_lists
from scaled_parameters import get_scaled_parameters

#%% Statistics on peaks and troughs without lineage
def compute_stats(dataset):
    peak_counter = Counter()
    trough_counter = Counter()
    peak_lengths = []
    trough_lengths = []
    for _, cell in load_dataset(dataset):
        for frame_data in cell:
            xs = frame_data["xs"]
            ys = frame_data["ys"]
            
            peaks = frame_data['peaks']
            troughs = frame_data['troughs']
            peak_counter[len(peaks)] += 1
            trough_counter[len(troughs)] += 1
            if len(peaks) + len(troughs) >= 2:
                peaks_dist, troughs_dist = compute_dist(peaks, troughs, xs)
                peak_lengths.extend(peaks_dist)
                trough_lengths.extend(troughs_dist)
    stats = {
        "peak_counter": peak_counter,
        "trough_counter": trough_counter,
        "peak_lengths": peak_lengths,
        "trough_lengths": trough_lengths,
    }
    return stats


def compute_dist(peaks, troughs, xs):
    peaks_dist = []
    troughs_dist = []
    for peak in peaks:
        score = 0
        right = False
        left = False
        if np.any(troughs > peak):
            pos = np.min(troughs[troughs > peak])
            score += np.abs(xs[pos] - xs[peak])
            right = True
        if np.any(troughs < peak):
            pos = np.max(troughs[troughs < peak])
            score += np.abs(xs[pos] - xs[peak])
            right = True
        if right and left:
            peaks_dist.append(score/2)
        else:
            peaks_dist.append(score)
    for trough in troughs:
        score = 0
        right = False
        left = False
        if np.any(peaks > trough):
            pos = np.min(peaks[peaks > trough])
            score += np.abs(xs[pos] - xs[trough])
            right = True
        if np.any(peaks < trough):
            pos = np.max(peaks[peaks < trough])
            score += np.abs(xs[pos] - xs[trough])
            right = True
        if right and left:
            troughs_dist.append(score/2)
        else:
            troughs_dist.append(score)
    return peaks_dist, troughs_dist 
        
        
        
    

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
        f"Distribution of the {feature} length ({dataset} | "
        f"{len(lengths)} {feature}s)"
    )
    density, bins = np.histogram(lengths , 40)
    spl = splrep(bins[1:], density, s=3*len(lengths), per=False)
    x2 = np.linspace(bins[0], bins[-1], 200)
    y2 = splev(x2, spl)
    y2[y2<0]=0
    ax.plot(x2, y2, '--', color = 'purple', label= 'smooth approximation', )
    ax.hist(lengths, 40,  color="grey") #density=True,
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



#%% Statistics on peaks and troughs with lineage

def feature_creation(dataset_names):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    bin_num = params["bin_number_hist_feat_crea"]
    smooth_param = params["smoothing_hist_feat_crea"]
    plt.figure()
    stat_list = []
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 4:
                        root = pnt_ROI[key][0]
                        time = pnt_list[root][-2]
                        if time >0 :
                            generation = int(pnt_list[root][1])
                            total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                            x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0])/total_length
                            if generation >2 :
                                if 0 <= x_coord <= 0.5:
                                    stat_list.append(x_coord)
                                else :
                                    stat_list.append(1-x_coord)
    
    stat_list = np.array(stat_list)
    density, bins = np.histogram(stat_list , bin_num)
    spl = splrep(bins[1:], density, s=smooth_param, per=False)
    x2 = np.linspace(bins[0], bins[-1], 200)
    y2 = splev(x2, spl)
    title = (
        f"Relative distribution of feature creation with dataset \'{dataset_names}\',\n and {len(stat_list)} features tracked"
    )
    plt.plot(x2, y2, 'r-', label='smooth approximation')
    plt.hist(stat_list, bin_num, color="grey")
    plt.xlabel(r'$\leftarrow$ pole | center $\rightarrow$ ')
    plt.legend()
    plt.title(title)
    plt.show()


def feature_creation_comparison(dataset_names1, dataset_names2):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names1 in params.keys():
        datasets1 = params[dataset_names1]
    else : 
        datasets1 = dataset_names1
    
    if dataset_names2 in params.keys():
        datasets2 = params[dataset_names2]
    else : 
        datasets2 = dataset_names2
    
    bin_num = params["bin_number_hist_feat_crea"]
    
    
    stat_list1 = []
    for dataset in datasets1:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 4:
                        root = pnt_ROI[key][0]
                        time = pnt_list[root][-2]
                        if time >0 :
                            generation = int(pnt_list[root][1])
                            total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                            x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0])/total_length
                            if generation >2 :
                                if 0 <= x_coord <= 0.5:
                                    stat_list1.append(x_coord)
                                else :
                                    stat_list1.append(1-x_coord)
    
    stat_list1 = np.array(stat_list1)
    
    stat_list2 = []
    for dataset in datasets2:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 4:
                        root = pnt_ROI[key][0]
                        time = pnt_list[root][-2]
                        if time >0 :
                            generation = int(pnt_list[root][1])
                            total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                            x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0])/total_length
                            if generation >2 :
                                if 0 <= x_coord <= 0.5:
                                    stat_list2.append(x_coord)
                                else :
                                    stat_list2.append(1-x_coord)
    stat_list2 = np.array(stat_list2)
    pvalue = stats.ttest_ind(stat_list1,stat_list2).pvalue
    
    plt.figure()
    plt.hist(stat_list1, bin_num, alpha=0.5, label=f' dataset {dataset_names1}, {len(stat_list1)} features', density=True)
    plt.hist(stat_list2, bin_num, alpha=0.5, label=f' dataset {dataset_names2}, {len(stat_list2)} features', density=True)
    plt.annotate(f"P={pvalue:.2e}",(0.4,3))
    plt.xlabel(r'$\leftarrow$ pole | center $\rightarrow$ ')
    plt.legend()
    plt.title("Comparision of feature creation distributions")

    plt.show()




def feature_displacement(dataset_names, plot=True, return_smooth_approx=False):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    smooth_param = params["smoothing_hist_feat_crea"]
    
    stat_list = []
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 6:
                        root = pnt_ROI[key][0]
                        leaf = pnt_ROI[key][-1]
                        time_diff = pnt_list[leaf][-2] - pnt_list[root][-2]
                        pos_diff = abs(pnt_list[leaf][3]-pnt_list[root][3])
                        generation = int(pnt_list[leaf][1])
                        total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                        x_coord = abs(pnt_list[leaf][3]-cell[generation]['xs'][0])/total_length
                        if 0 <= x_coord <= 0.5:
                            stat_list.append([x_coord,pos_diff/time_diff])
                        else :
                            stat_list.append([1-x_coord,pos_diff/time_diff])
    
    stat_list = np.transpose(np.array(stat_list))
    sort = np.argsort(stat_list[0,:])
    stat_list = stat_list[:,sort]
    
    spl = splrep(stat_list[0,:], stat_list[1,:], s=smooth_param, per=False)
    x2 = np.linspace(stat_list[0,0], stat_list[0,-1], 200)
    y2 = splev(x2, spl)
    if return_smooth_approx:
        return x2, y2
    title = (
        f"Drift of features with dataset \'{dataset_names}\',\n and {len(stat_list[0,:])} features tracked"
    )
    if plot:
        plt.figure()
        plt.scatter(stat_list[0,:], stat_list[1,:])
        plt.plot(x2, y2, 'r-', label='smooth approximation')
        plt.xlabel(r'$\leftarrow$ pole | center $\rightarrow$ ')
        plt.ylabel(r'Drift speed $\mu m (min)^{-1}$')
        plt.legend()
        plt.title(title)
        
        plt.figure()
        plt.plot(x2, y2, 'r-', label='smooth approximation')
        plt.xlabel(r'$\leftarrow$ pole | center $\rightarrow$ ')
        plt.ylabel(r'Drift speed $\mu m (min)^{-1}$')
        plt.title(title)
        plt.legend()
        
        plt.show()            
                    
                


def feature_displacement_comparison(*dataset_name_list):
    plt.figure()
    for dataset_names in dataset_name_list:
        x2, y2 = feature_displacement(dataset_names, plot=False, return_smooth_approx=True)
        plt.plot(x2, y2, label=f'{dataset_names}')
    title = (
        f"Drift speed of features comparison between datasets \n {dataset_name_list}"
    )
    plt.xlabel(r'$\leftarrow$ pole | center $\rightarrow$ ')
    plt.legend()
    plt.title(title)
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
    # main()
    feature_creation("all")#"WT_mc2_55/30-03-2015", "all""no_WT"
    feature_creation_comparison("WT_drug",'WT_no_drug')
    feature_displacement("WT")
    feature_displacement_comparison("no_WT","WT_drug",'WT_no_drug')