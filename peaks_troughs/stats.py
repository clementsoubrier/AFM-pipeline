import glob
import os
import sys
from collections import Counter
import statistics

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import splev, splrep


package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from peaks_troughs.group_by_cell import Orientation, load_dataset, get_peak_troughs_lineage_lists
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
    
    stat_list = []
    stat_list_relat = []
    length_list = []
    stat_list_np = []
    stat_list_op = []
    
    
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 4:
                        root = pnt_ROI[key][0]
                        frame_data = cell[0] #pnt_ROI[key][1]
                        orientation = Orientation(frame_data['orientation'])
                        time = pnt_list[root][-2]
                        if time >0 :
                            generation = int(pnt_list[root][1])
                            if generation >2 :
                                total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                                length_list.append(total_length)
                                x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0])
                                match orientation:
                                    case Orientation.UNKNOWN:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            
                                    case Orientation.NEW_POLE_OLD_POLE:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            stat_list_op.append(x_coord)
                                        else:
                                            stat_list_np.append(x_coord)
                                            
                                    case Orientation.OLD_POLE_NEW_POLE:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            stat_list_np.append(x_coord)
                                        else:
                                            stat_list_op.append(x_coord)
                                stat_list.append(x_coord)
                                stat_list_relat.append(x_coord/total_length)
                                
                                
                                
    
    stat_list = np.array(stat_list)
    stat_list_relat = np.array(stat_list_relat)
    length_list = np.array(length_list)
    stat_list_op = np.array(stat_list_op)
    stat_list_np = np.array(stat_list_np)
    
    # density, bins = np.histogram(stat_list , bin_num)
    # spl = splrep(bins[1:], density, s=smooth_param, per=False)
    # x2 = np.linspace(bins[0], bins[-1], 200)
    # y2 = splev(x2, spl)
    
    plt.figure()
    title = (
        f"Distribution of feature creation with dataset \'{dataset_names}\',\n and {len(stat_list)} features tracked"
    )
    # plt.plot(x2, y2, 'r-', label='smooth approximation')
    plt.hist(stat_list, bin_num, color="grey", alpha=0.6, label= 'feature creation')
    plt.hist(length_list, bin_num, color="r", alpha=0.6, label= 'total length')
    plt.xlabel(r'Distance to the pole ($\mu m$) ')
    plt.legend()
    plt.title(title)
    plt.show()
    
    plt.figure()
    title = (
        f"Feature creation with dataset \'{dataset_names}\',\n and {len(stat_list_relat)} features tracked, normalized total lenght"
    )
    plt.hist(stat_list_relat, bin_num, color="grey", alpha=0.6, label= 'feature creation')
    plt.xlabel(r' $\leftarrow \; \text{pole}\;|\; \text{center}\; \rightarrow$ ')
    plt.legend()
    plt.title(title)
    plt.show()
    
    
    plt.figure()
    title = (
        f"Distribution of feature creation with dataset \'{dataset_names}\',\n and {len(stat_list_op)+len(stat_list_np)} features tracked"
    )
    res = stats.kstest(stat_list_op,stat_list_np)
    print(res.statistic, res.pvalue)
    plt.hist(stat_list_op, bin_num, color="g", alpha=0.6,  label= 'old pole')
    plt.hist(stat_list_np, bin_num, color="b", alpha=0.6, label= 'new pole')
    plt.annotate(f'pvalue = {res.pvalue:.2e}', (3,15))
    plt.xlabel(r'Distance to the pole ($\mu m$) ')
    plt.legend()
    plt.title(title)
    plt.show()



def peaks_trough_diff(dataset_names):
    
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    bin_num = params["bin_number_hist_feat_crea"]
    stat_list = []
    position_list = []
    diff_list = []
    
    for dataset in datasets:
        for _, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                for frame_data in cell:
                    ys = frame_data["ys"]
                    xs = frame_data["xs"]-frame_data["xs"][0]
                    peaks = frame_data["peaks"]
                    troughs = frame_data["troughs"]
                    totlen = len(peaks)+len(troughs)
                    if totlen>=2:
                        feature = np.zeros(totlen, dtype=int)
                        x_coord = np.zeros(totlen)
                        if peaks[0]<troughs[0]:
                            feature[::2] = ys[peaks]
                            feature[1::2] = ys[troughs]
                            x_coord[::2] = xs[peaks]
                            x_coord[1::2] = xs[troughs]
                        else:
                            feature[1::2] = ys[peaks]
                            feature[::2] = ys[troughs]
                            x_coord[1::2] = xs[peaks]
                            x_coord[::2] = xs[troughs]
                        stat_list.append(np.absolute(feature[:-1]-feature[1:]))
                        diff_list.append(x_coord[1:]-x_coord[:-1])
                        for elem in x_coord[:-1]:
                            if elem < xs[-1]/2:
                                position_list.append(elem/xs[-1])
                            else :
                                position_list.append((xs[-1]-elem)/xs[-1])

    
    stat_list = np.concatenate(stat_list)
    position_list = np.array(position_list)
    diff_list = np.concatenate(diff_list)
         
    good_ind = (stat_list<500) & (diff_list<4) & (position_list<10)
    stat_list = stat_list[good_ind]
    position_list = position_list[good_ind]
    diff_list = diff_list[good_ind]
    
    
    plt.figure()
    title = (
        f"Distribution of feature height with dataset \'{dataset_names}\',\n and {len(stat_list)} individual features"
    )
    plt.hist(stat_list, bin_num, color="grey")
    plt.xlabel(r'Height ($n m$) ')
    plt.vlines(np.average(stat_list),0,8000, label=f'average : {np.average(stat_list):.2e}', color = 'b')
    plt.vlines(np.median(stat_list),0,8000, label=f'median : {np.median(stat_list):.2e}', color = 'r')
    plt.legend()
    plt.title(title)
    
    plt.figure()
    title = (
        f"Feature height with dataset \'{dataset_names}\',\n and {len(stat_list)} individual features, normalized total lenght"
    )
    # plt.scatter(position_list, stat_list, marker='.')
    bins = 10
    data_list = []
    for elem in range(bins):
        data_list.append([])
    for i, elem in enumerate(position_list):
        index = int(elem *2*bins)
        if index == 10:
            index = 9
        data_list[index].append(stat_list[i])

    plt.boxplot(data_list,labels= np.linspace(0.5,5,10)/10, showfliers=False)       #, showmeans=True, meanline=True
    # slope, intercept = statistics.linear_regression(position_list, stat_list)
    # min_x= np.min(position_list)
    # max_x= np.max(position_list)
    # plt.plot([min_x,max_x], [slope*min_x+intercept, slope*max_x+intercept], color='r', label='linear interpolation')
    plt.xlabel(r' $\leftarrow \;\text{pole}\;|\;  \text{center} \;\rightarrow$')
    plt.ylabel(r'Height ($n m$)')
    # plt.legend()
    plt.title(title)
    
    plt.figure()
    title = (
        f"Distribution of feature length with dataset \'{dataset_names}\',\n and {len(stat_list)} individual features"
    )
    plt.hist(diff_list, bin_num, color="grey")
    plt.xlabel(r'Lenght ($\mu m$) ')
    plt.vlines(np.average(diff_list),0,7000, label=f'average : {np.average(diff_list):.2e}', color = 'b')
    plt.vlines(np.median(diff_list),0,7000, label=f'median : {np.median(diff_list):.2e}', color = 'r')
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
    pole_size = params["pole_region_size"]
    
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
                        generation = int(pnt_list[root][1]) #leaf
                        total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                        x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0]) #/total_length #leaf
                        if x_coord > 0.5 * total_length:
                            x_coord = total_length - x_coord

                        if time_diff >= 50:
                            stat_list.append([x_coord,pos_diff/time_diff])
    
    stat_list = np.array(stat_list)
    
    # stat_list = np.transpose(np.array(stat_list))
    # sort = np.argsort(stat_list[0,:])
    # stat_list = stat_list[:,sort]
    
    # spl = splrep(stat_list[0,:], stat_list[1,:], s=smooth_param, per=False)
    # x2 = np.linspace(stat_list[0,0], stat_list[0,-1], 200)
    # y2 = splev(x2, spl)
    # if return_smooth_approx:
    #     return x2, y2
    title = (
        f"Variation of feature position with dataset \'{dataset_names}\',\n and {len(stat_list[:,0])} features tracked"
    )
    if plot:
        plt.figure()
        # plt.scatter(stat_list[0,:], stat_list[1,:])
        # plt.plot(x2, y2, 'r-', label='smooth approximation')
        mask1 = stat_list[:,0] <= pole_size
        mask2 = stat_list[:,0] > pole_size
        val1 = stat_list[mask1]
        val2 = stat_list[mask2]
        plt.hist(val1[:,1], color='gray', alpha=0.5, label = 'Near pole region', density=True, bins = 30)
        plt.hist(val2[:,1], color='red', alpha=0.5, label = 'Near center region', density=True)
        plt.xlabel(r'Variation of feature position $\mu m (min)^{-1}$')
        plt.legend()
        plt.title(title)
        
        
        plt.show()            
                    
                
def feature_len_height_variation(dataset_names):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    pole_size = params["pole_region_size"]
    stat_list_len = []
    stat_list_height = []
    
    
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 6:
                        root = pnt_ROI[key][0]
                        leaf = pnt_ROI[key][-1]
                        feature_type = bool(pnt_list[root][2])
                        feature = feature_type * 'peaks' + (1-feature_type) * 'troughs'
                        non_feature = (1-feature_type) * 'peaks' + feature_type * 'troughs'
                        time_diff = pnt_list[leaf][-2] - pnt_list[root][-2]
                        if time_diff >= 50:
                            generation_root = int(pnt_list[root][1]) 
                            generation_leaf = int(pnt_list[leaf][1]) 
                            
                            total_length =  abs(cell[generation_root]['xs'][0]-cell[generation_root]['xs'][-1])
                            x_coord = abs(pnt_list[root][3]-cell[generation_root]['xs'][0]) 
                            if x_coord > 0.5 * total_length:
                                x_coord = total_length - x_coord
                            
                            ind_root = np.nonzero(cell[generation_root][feature + '_index'] == root)[0][0]
                            root_pos = cell[generation_root][feature][ind_root]
                            
                            ind_leaf = np.nonzero(cell[generation_leaf][feature + '_index'] == leaf)[0][0]
                            leaf_pos = cell[generation_leaf][feature][ind_leaf]
                            
                            pos_diff_1l = 0
                            pos_diff_2l = 0
                            pos_diff_1r = 0
                            pos_diff_2r = 0
                            
                            mask_r = cell[generation_root][non_feature]<root_pos
                            mask_l = cell[generation_leaf][non_feature]<leaf_pos
                            
                            if np.any(mask_r) and np.any(mask_l):
                                    left_root = cell[generation_root][non_feature][np.nonzero(mask_r)[0][-1]]
                                    left_leaf = cell[generation_leaf][non_feature][np.nonzero(mask_l)[0][-1]]
                                    pos_diff_1l = cell[generation_leaf]['xs'][leaf_pos] - cell[generation_leaf]['xs'][left_leaf]\
                                                    - cell[generation_root]['xs'][root_pos] + cell[generation_root]['xs'][left_root] 
                                    # stat_list_len.append([x_coord, abs(pos_diff_1l)/time_diff]) 
                                    
                                    pos_diff_2l = cell[generation_leaf]['ys'][leaf_pos] - cell[generation_leaf]['ys'][left_leaf]\
                                                    - cell[generation_root]['ys'][root_pos] + cell[generation_root]['ys'][left_root] 
                                    # stat_list_height.append([x_coord, abs(pos_diff_2l)/time_diff])
                                    
                            mask_r = cell[generation_root][non_feature]>root_pos
                            mask_l = cell[generation_leaf][non_feature]>leaf_pos
                            
                            if np.any(mask_r) and np.any(mask_l):
                                    right_root = cell[generation_root][non_feature][np.nonzero(mask_r)[0][-1]]
                                    right_leaf = cell[generation_leaf][non_feature][np.nonzero(mask_l)[0][-1]]
                                    pos_diff_1r =  cell[generation_leaf]['xs'][leaf_pos] - cell[generation_leaf]['xs'][right_leaf]\
                                                    - cell[generation_root]['xs'][root_pos] + cell[generation_root]['xs'][right_root] 
                                    # stat_list_len.append([x_coord, abs(pos_diff_1r)/time_diff])
                                    
                                    pos_diff_2r = cell[generation_leaf]['ys'][leaf_pos] - cell[generation_leaf]['ys'][right_leaf]\
                                                    - cell[generation_root]['ys'][root_pos] + cell[generation_root]['ys'][right_root] 
                                    # stat_list_height.append([x_coord, abs(pos_diff_2r)/time_diff])
                                    
                            if pos_diff_1l>0 + pos_diff_1r>0 == 2:

                                stat_list_len.append([x_coord,abs(pos_diff_1l-pos_diff_1r)/2 /time_diff]) 
                                stat_list_height.append([x_coord,abs(pos_diff_2l-pos_diff_2r)/2 /time_diff])
                                   
                            else :
                                if pos_diff_1l>0:
                                    stat_list_len.append([x_coord,abs(pos_diff_1l) /time_diff]) 
                                else:
                                    stat_list_len.append([x_coord,abs(pos_diff_1r) /time_diff])
                                if pos_diff_2l>0:
                                    stat_list_height.append([x_coord,abs(pos_diff_2l) /time_diff]) 
                                else:
                                    stat_list_height.append([x_coord,abs(pos_diff_2r) /time_diff])
                                
                            
    
    
    stat_list_len = np.array(stat_list_len)
    stat_list_height = np.array(stat_list_height)

    title = (
        f"Variation of inter-feature distance with dataset \'{dataset_names}\',\n and {len(stat_list_len[:,0])} features tracked"
    )

    plt.figure()
    mask1 = stat_list_len[:,0] <= pole_size
    mask2 = stat_list_len[:,0] > pole_size
    val1 = stat_list_len[mask1]
    val2 = stat_list_len[mask2]
    plt.hist(val1[:,1], color='gray', alpha=0.5, label = 'Near pole region', density=True)
    plt.hist(val2[:,1], color='red', alpha=0.5, label = 'Near center region', density=True)
    plt.xlabel(r'Variation of inter-feature distance $\mu m (min)^{-1}$')
    plt.legend()
    plt.title(title)
    
    title = (
        f"Variation of inter-feature amplitude with dataset \'{dataset_names}\',\n and {len(stat_list_len[:,0])} features tracked"
    )
    plt.figure()
    mask1 = stat_list_height[:,0] <= pole_size
    mask2 = stat_list_height[:,0] > pole_size
    val1 = stat_list_height[mask1]
    val2 = stat_list_height[mask2]
    plt.hist(val1[:,1], color='gray', alpha=0.5, label = 'Near pole region', density=True)
    plt.hist(val2[:,1], color='red', alpha=0.5, label = 'Near center region', density=True)
    plt.xlabel(r'Variation of inter-feature amplitude $n m (min)^{-1}$')
    plt.legend()
    plt.title(title)
    
    
    plt.show()         

# def feature_displacement_comparison(*dataset_name_list):
#     plt.figure()
#     for dataset_names in dataset_name_list:
#         x2, y2 = feature_displacement(dataset_names, plot=False, return_smooth_approx=True)
#         plt.plot(x2, y2, label=f'{dataset_names}')
#     title = (
#         f"Drift speed of features comparison between datasets \n {dataset_name_list}"
#     )
#     plt.xlabel(r'$\leftarrow$ pole | center $\rightarrow$ ')
#     plt.legend()
#     plt.title(title)
#     plt.show()    
    
   





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
    # feature_creation('all')     #"WT_mc2_55/30-03-2015", "all""no_WT"
    # peaks_trough_diff('all')
    # feature_creation_comparison("WT_drug",'WT_no_drug')
    # feature_displacement("WT_mc2_55/30-03-2015") #"all"
    feature_len_height_variation ("WT_mc2_55/30-03-2015")
    # feature_displacement_comparison("no_WT","WT_drug",'WT_no_drug')
# %%
