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


def compute_dist(peaks, troughs, xys, array=False): #can use xs or ys for lenght or height
    peaks_dist = []
    troughs_dist = []
    for peak in peaks:
        score = 0
        right = False
        left = False
        if np.any(troughs > peak):
            pos = np.min(troughs[troughs > peak])
            score += np.abs(xys[pos] - xys[peak])
            right = True
        if np.any(troughs < peak):
            pos = np.max(troughs[troughs < peak])
            score += np.abs(xys[pos] - xys[peak])
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
            score += np.abs(xys[pos] - xys[trough])
            right = True
        if np.any(peaks < trough):
            pos = np.max(peaks[peaks < trough])
            score += np.abs(xys[pos] - xys[trough])
            right = True
        if right and left:
            troughs_dist.append(score/2)
        else:
            troughs_dist.append(score)
    if array:
        return np.array(peaks_dist), np.array(troughs_dist)
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

# Spatial distribution of feature creation 
def feature_creation(dataset_names):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    bin_num = params["bin_number_hist_feat_crea"]
    smooth_param = params["smoothing_hist_feat_crea"]
    
    stat_list = {'peaks':[], 'troughs':[]}
    stat_list_relat = []
    length_list = []
    stat_list_np = {'peaks':[], 'troughs':[]}
    stat_list_op = {'peaks':[], 'troughs':[]}
    
    
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
                                feat_ind = int(pnt_list[root][2])
                                feature = feat_ind*'peaks' + (1-feat_ind)*'troughs'
                                match orientation:
                                    case Orientation.UNKNOWN:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            
                                    case Orientation.NEW_POLE_OLD_POLE:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            stat_list_op[feature].append(x_coord)
                                        else:
                                            stat_list_np[feature].append(x_coord)
                                            
                                    case Orientation.OLD_POLE_NEW_POLE:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            stat_list_np[feature].append(x_coord)
                                        else:
                                            stat_list_op[feature].append(x_coord)
                                stat_list[feature].append(x_coord)
 
                                stat_list_relat.append(x_coord/total_length)
                                
                                
                                
    
    
    
    
    plt.figure()
    title = (
        f"Distribution of feature creation with dataset \'{dataset_names}\',\n and {len(stat_list['peaks']+stat_list['troughs'])} features tracked"
    )
    plt.hist(stat_list['peaks']+stat_list['troughs'], bin_num, color="grey", alpha=0.6, label= 'feature creation', density=True)
    plt.hist(length_list, bin_num, color="r", alpha=0.6, label= 'total length', density=True)
    plt.xlabel(r'Distance to the pole ($\mu m$) ')
    plt.legend()
    plt.title(title)
    
    plt.figure()
    title = (
        f"Distribution of feature creation with dataset \'{dataset_names}\',\n and {len(stat_list['peaks']+stat_list['troughs'])} features tracked"
    )
    res = stats.kstest(stat_list['peaks'], stat_list['troughs'])
    print(res.statistic, res.pvalue)
    plt.hist(stat_list['peaks'], bin_num, color="g", alpha=0.6,  label= 'peaks', density=True)
    plt.hist(stat_list['troughs'], bin_num, color="b", alpha=0.6, label= 'troughs', density=True)
    plt.annotate(f'pvalue = {res.pvalue:.2e}', (3,0.6))
    plt.xlabel(r'Distance to the pole ($\mu m$) ')
    plt.legend()
    plt.title(title)

    
    plt.figure()
    title = (
        f"Feature creation with dataset \'{dataset_names}\',\n and {len(stat_list_relat)} features tracked, normalized total lenght"
    )
    plt.hist(stat_list_relat, bin_num, color="grey", alpha=0.6, label= 'feature creation')
    plt.xlabel(r' $\leftarrow \; \text{pole}\;|\; \text{center}\; \rightarrow$ ')
    plt.legend()
    plt.title(title)

    
    _, ax = plt.subplots()
    ax.boxplot([stat_list_np['peaks'],
                stat_list_op['peaks'],
                stat_list_np['troughs'],
                stat_list_op['troughs']]
               ,  showfliers=False, showmeans=True, meanline=True) 
    ax.set_xticklabels([f"Peaks new pole \n creation", f"Peaks old pole \n creation", f"Troughs new pole \n creation", f"Troughs old pole \n creation"])
    ax.set_ylabel(r"Distance to the pole ($\mu m$)")
    pvalue1 = stats.ttest_ind(stat_list_np['peaks'], stat_list_op['peaks']).pvalue
    x1 = 1
    x2 = 2 
    y = 2.8
    h=0.1
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    pvalue1 = stats.ttest_ind(stat_list_np['peaks'], stat_list_np['troughs']).pvalue
    x1 = 1
    x2 = 3 
    y = 3.1
    h=0.1
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    pvalue1 = stats.ttest_ind(stat_list_op['peaks'], stat_list_op['troughs']).pvalue
    x1 = 2
    x2 = 4
    y = 2.3
    h=0.1
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    pvalue1 = stats.ttest_ind(stat_list_np['troughs'], stat_list_op['troughs']).pvalue
    x1 = 3
    x2 = 4 
    y = 2
    h=0.1
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    plt.show()



#comparision of old pole / new pole distributions
def feature_properties_pole(dataset_names):
    params = get_scaled_parameters(data_set=True, stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    pole_size = params["pole_region_size"] 
    whole_stat = {}
    for prop in ['lenght', 'height']:
        whole_stat[prop] = {}
        for pole in ['np', 'op']:
            whole_stat[prop][pole]={}
            for feat in ['peaks', 'troughs']:
                whole_stat[prop][pole][feat] = []
    
    
    for dataset in datasets:
        
        for _, cell in load_dataset(dataset, False):
            stat = {}
            for prop in ['lenght', 'height']:
                stat[prop] = {}
                for pole in ['np', 'op']:
                    stat[prop][pole]={}
                    for feat in ['peaks', 'troughs']:
                        stat[prop][pole][feat] = []
            if len(cell) > 1:
                orientation = Orientation(cell[0]['orientation'])
                if orientation == Orientation.UNKNOWN:
                    continue
                for frame_data in cell:
                    if len(frame_data['peaks']) >=1 and len(frame_data['troughs'])>=1:
                        height = {}
                        length = {}
                        height['peaks'], height['troughs'] = compute_dist(frame_data['peaks'], frame_data['troughs'], frame_data['ys'],array=True)
                        length['peaks'], length['troughs']= compute_dist(frame_data['peaks'], frame_data['troughs'], frame_data['xs'],array=True)
                        for feature in ['peaks', 'troughs']: 
                            mask = frame_data['xs'][frame_data[feature]]- frame_data['xs'][0] <= pole_size
                            match orientation:
                                case Orientation.NEW_POLE_OLD_POLE:
                                    pole = 'np'
                                case Orientation.OLD_POLE_NEW_POLE:
                                    pole = 'op'
                            if np.any(mask):
                                        stat['lenght'][pole][feature].extend(list(length[feature][mask])) 
                                        stat['height'][pole][feature].extend(list(height[feature][mask]))
                            mask = frame_data['xs'][frame_data[feature]] >= frame_data['xs'][-1]-pole_size
                            match orientation:
                                case Orientation.NEW_POLE_OLD_POLE:
                                    pole = 'op'
                                case Orientation.OLD_POLE_NEW_POLE:
                                    pole = 'np'
                            if np.any(mask):
                                        stat['lenght'][pole][feature].extend(list(length[feature][mask])) 
                                        stat['height'][pole][feature].extend(list(height[feature][mask]))
            for prop in ['lenght', 'height']:
                    for pole in ['np', 'op']:
                        for feat in ['peaks', 'troughs']:
                            l = stat[prop][pole][feat]
                            if len(l)>0:
                                whole_stat[prop][pole][feat].append(np.average(np.array(l)))

                    
                                

    numbers = [len(whole_stat['height'][pole][feat])for pole in ['np', 'op']for feat in ['peaks', 'troughs']]
    _, ax = plt.subplots()
    ax.boxplot([whole_stat['lenght']['np']['peaks'],
                whole_stat['lenght']['op']['peaks'],
                whole_stat['lenght']['np']['troughs'],
                whole_stat['lenght']['op']['troughs']]
               ,  showfliers=False, showmeans=True, meanline=True) 
    ax.set_xticklabels(["Peaks new pole", "Peaks old pole", "Troughs new pole", "Troughs old pole"])
    ax.set_ylabel(r"Distance ($\mu m$)")
    ax.set_title(f"Inter-feature distance with dataset \'{dataset_names}\',\n and {numbers} cells")
    pvalue1 = stats.ttest_ind(whole_stat['lenght']['np']['peaks'], whole_stat['lenght']['op']['peaks']).pvalue
    x1 = 1
    x2 = 2 
    y = 2.5
    h=0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    pvalue2 = stats.ttest_ind(whole_stat['lenght']['np']['troughs'], whole_stat['lenght']['op']['troughs']).pvalue
    x1 = 3
    x2 = 4 
    y = 3.2
    h=0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue2:.2e} ", ha='center', va='bottom')
    pvalue1 = stats.ttest_ind(whole_stat['lenght']['np']['peaks'], whole_stat['lenght']['np']['troughs']).pvalue
    x1 = 1
    x2 = 3 
    y = 2.85
    h=0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    pvalue2 = stats.ttest_ind(whole_stat['lenght']['op']['troughs'], whole_stat['lenght']['op']['peaks']).pvalue
    x1 = 2
    x2 = 4 
    y = 3.55
    h=0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue2:.2e} ", ha='center', va='bottom')
    
    
    
    _, ax = plt.subplots()
    ax.boxplot([whole_stat['height']['np']['peaks'],
                whole_stat['height']['op']['peaks'],
                whole_stat['height']['np']['troughs'],
                whole_stat['height']['op']['troughs']]
               ,  showfliers=False, showmeans=True, meanline=True) 
    ax.set_xticklabels(["Peaks new pole", "Peaks old pole", "Troughs new pole", "Troughs old pole"])
    ax.set_ylabel(r"Amplitude ($n m$)")
    ax.set_title(f"Inter-feature amplitude with dataset \'{dataset_names}\',\n and {numbers} cells")
    pvalue1 = stats.ttest_ind(whole_stat['height']['np']['peaks'], whole_stat['height']['op']['peaks']).pvalue
    x1 = 1
    x2 = 2 
    y = 360
    h = 20
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    pvalue2 = stats.ttest_ind(whole_stat['height']['np']['troughs'], whole_stat['height']['op']['troughs']).pvalue
    x1 = 3
    x2 = 4 
    y = 480
    h = 20
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue2:.2e} ", ha='center', va='bottom')
    pvalue1 = stats.ttest_ind(whole_stat['height']['np']['peaks'], whole_stat['height']['np']['troughs']).pvalue
    x1 = 1
    x2 = 3 
    y = 520
    h=20
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    pvalue2 = stats.ttest_ind(whole_stat['height']['op']['troughs'], whole_stat['height']['op']['peaks']).pvalue
    x1 = 2
    x2 = 4 
    y = 580
    h=20
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue2:.2e} ", ha='center', va='bottom')
    
    plt.show()
                                

def feature_number(dataset_names):
    params = get_scaled_parameters(data_set=True, paths_and_names=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    
    peak_root = []
    peak_leaf = []
    trough_root = []
    trough_leaf = []
    count = 0
    
    for dataset in datasets:
        data_direc = params["main_data_direc"]
        roi_dic_name = params["roi_dict_name"]
        roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()
        
        for roi_id, cell in load_dataset(dataset, False):
            
            if len(cell) > 5:
                if roi_dic[roi_id]['Parent'] != '' and len(roi_dic[roi_id]['Children'])>0:
                    peak_root.append(len(cell[0]['peaks']))
                    trough_root.append(len(cell[0]['troughs']))
                    peak_leaf.append(len(cell[-1]['peaks']))
                    trough_leaf.append(len(cell[-1]['troughs']))
                    count += 1
                elif  roi_dic[roi_id]['Parent'] != '':
                    peak_root.append(len(cell[0]['peaks']))
                    trough_root.append(len(cell[0]['troughs']))
                    count += 1
                    
                elif len(roi_dic[roi_id]['Children'])>0 :
                    peak_leaf.append(len(cell[-1]['peaks']))
                    trough_leaf.append(len(cell[-1]['troughs']))
                    count += 1
                    
                
    _, ax = plt.subplots()
    ax.boxplot([peak_root, peak_leaf, trough_root, trough_leaf], showfliers=False, showmeans=True, meanline=True)
    ax.set_xticklabels([f"Peaks \n after division", f"Peaks \n before division", f"Troughs \n after division", f"Troughs \n before division"])
    ax.set_ylabel("Feature number")
    ax.set_title(f"Number of features with dataset \'{dataset_names}\',\n and {count} cells")
    pvalue1 = stats.ttest_ind(peak_root, peak_leaf).pvalue
    pvalue2 = stats.ttest_ind( trough_root, trough_leaf).pvalue
    x1 = 1
    x2 = 2 
    y = 6.2
    h=0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ***", ha='center', va='bottom')
    
    x1 = 3
    x2 = 4 
    y = 6.2
    h=0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue2:.2e} ***", ha='center', va='bottom')
    
    

def feature_general_properties(dataset_names):
    
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
         
    good_ind = (stat_list<500)#& (diff_list<4) & (position_list<10)
    stat_list = stat_list[good_ind]
    position_list = position_list[good_ind]
    diff_list = diff_list[good_ind]
    
    
    fig, ax = plt.subplots(1,2)
    title = (
        f" dataset \'{dataset_names}\',\n and {len(stat_list)} individual features"
    )
    plt.title(title)
    ax[0].boxplot(stat_list, showfliers=False, showmeans=True, meanline=True) 
    ax[0].set_xticklabels(['Inter-feature amplitude'])
    ax[0].set_ylabel(r'($n m$) ')

    
    ax[1].boxplot(diff_list, showfliers=False, showmeans=True, meanline=True) 
    ax[1].set_xticklabels(['Inter-feature distance'])
    ax[1].set_ylabel(r'($\mu m$) ')
    
    
    
    
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
    
    
    plt.show()
    
    return stat_list, position_list, diff_list
    
                    
                    

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

def get_feature_ind(feature_id, cell, pnt_list):
    generation = int(pnt_list[feature_id][1])
    if pnt_list[feature_id][2] == 1:
        ind = np.argmax(cell[generation]['peaks_index']==feature_id)
        res = cell[generation]['peaks'][ind]
    else :
        ind = np.argmax(cell[generation]['troughs_index']==feature_id)
        res = cell[generation]['troughs'][ind]
    return res, generation


def feature_displacement(dataset_names, plot=True, return_smooth_approx=False):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    pole_size = params["pole_region_size"]
    
    stat_list_p = []
    stat_list_c = []
    height_list_p = []
    height_list_c = []
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 6:
                        pos_list = []
                        height_list = []
                        time_list = []
                        len_list = []
                        for elem in pnt_ROI[key]:
                            ind, generation = get_feature_ind(elem, cell, pnt_list)
                            pos_list.append(cell[generation]['xs'][ind])
                            height_list.append(cell[generation]['ys'][ind])
                            time_list.append(cell[generation]['timestamp'])
                            len_list.append([cell[generation]['xs'][0], cell[generation]['xs'][-1]])
                            
                        pos_list = np.array(pos_list)
                        height_list = np.array(height_list )
                        time_list = np.array(time_list)
                        len_list = np.array(len_list)
                        
                        mask = (pos_list - len_list[:,0]  <= pole_size) | (pos_list >= len_list[:,1]-pole_size)
                        if np.sum(mask)>=2:
                            last_in_pole = np.argwhere(mask)[-1][0]
                            if time_list[last_in_pole]- time_list[0]>10:
                                stat_list_p.append(abs(pos_list[last_in_pole]-pos_list[0])/(time_list[last_in_pole]- time_list[0]))
                                height_list_p.append(abs(height_list[last_in_pole]-height_list[0])/(time_list[last_in_pole]- time_list[0]))
                            if last_in_pole < len(mask) -1 :
                                if  time_list[-1] - time_list[last_in_pole]>10:
                                    stat_list_c.append(abs(pos_list[-1] - pos_list[last_in_pole])/(time_list[-1] - time_list[last_in_pole]))
                                    height_list_c.append(abs(height_list[-1] - height_list[last_in_pole])/(time_list[-1] - time_list[last_in_pole]))
                        else :
                            if  time_list[-1] - time_list[0]>10:
                                stat_list_c.append(abs(pos_list[-1] - pos_list[0])/(time_list[-1] - time_list[0]))
                                height_list_c.append(abs(height_list[-1] - height_list[0])/(time_list[-1] - time_list[0]))
                            
                            
                            
                            
                            
                            
                        # root = pnt_ROI[key][0]
                        # leaf = pnt_ROI[key][-1]
                        # time_diff = pnt_list[leaf][-2] - pnt_list[root][-2]
                        # pos_diff = abs(pnt_list[leaf][3]-pnt_list[root][3])
                        
                        # generation = int(pnt_list[leaf][1]) #root
                        # total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                        # x_coord = abs(pnt_list[leaf][3]-cell[generation]['xs'][0]) #root
                        
                        # if time_diff >= 10:
                        #     if (x_coord <= pole_size) | (x_coord >= total_length-pole_size):
                        #         stat_list_p.append(pos_diff/time_diff)
                        #     else :
                        #         stat_list_c.append(pos_diff/time_diff)
                                

    

    
    stat_list_p = np.array(stat_list_p)
    stat_list_c = np.array(stat_list_c)
    
    height_list_p = np.array(height_list_p)
    height_list_c = np.array(height_list_c)
    

    title = (
    f"Variation of feature position and height with dataset \'{dataset_names}\',\n and {len(stat_list_p),len(stat_list_c) } features tracked"
)
    _, ax = plt.subplots(1,2)
    ax[0].boxplot([stat_list_p*1000,stat_list_c*1000], widths = 0.5,  showfliers=False, showmeans=True, meanline=True) 
    ax[0].set_xticklabels(['Near pole', 'Near center'])
    ax[0].set_ylabel(f'Variation of feature : \n position '+r'$n m (min)^{-1}$')
    # ax[0].set_title(title)
    pvalue1 = stats.ttest_ind(stat_list_p,stat_list_c).pvalue
    x1 = 1
    x2 = 2 
    y = 10
    h = 0.5
    ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax[0].text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    
    
    
    title = (
    f"Variation of feature height with dataset \'{dataset_names}\',\n and {len(stat_list_p),len(stat_list_c) } features tracked"
)
    # _, ax = plt.subplots()
    ax[1].boxplot([height_list_p,height_list_c], widths = 0.5,  showfliers=False, showmeans=True, meanline=True) 
    ax[1].set_xticklabels(['Near pole', 'Near center'])
    ax[1].set_ylabel(r'height $nm (min)^{-1}$')
    # ax.set_title(title)
    pvalue1 = stats.ttest_ind(height_list_p,height_list_c).pvalue
    x1 = 1
    x2 = 2 
    y = 1.42
    h = 0.05
    ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax[1].text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')




    
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
                    if len(pnt_ROI[key]) >= 2:
                        root = pnt_ROI[key][0]
                        leaf = pnt_ROI[key][-1]
                        time_diff = pnt_list[leaf][-2] - pnt_list[root][-2]
                        if time_diff >= 50:
                            feature_type = bool(pnt_list[root][2])
                            feature = feature_type * 'peaks' + (1-feature_type) * 'troughs'
                            non_feature = (1-feature_type) * 'peaks' + feature_type * 'troughs'
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
                                    right_root = cell[generation_root][non_feature][np.nonzero(mask_r)[0][0]]
                                    right_leaf = cell[generation_leaf][non_feature][np.nonzero(mask_l)[0][0]]
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

    _, ax = plt.subplots()
    mask1 = stat_list_len[:,0] <= pole_size
    mask2 = stat_list_len[:,0] > pole_size
    val1 = stat_list_len[mask1]
    val2 = stat_list_len[mask2]
    ax.boxplot([val1[:,1],val2[:,1]],  showfliers=False, showmeans=True, meanline=True) 
    ax.set_xticklabels(['Near pole region', 'Near center region'])
    ax.set_ylabel(r'Variation of inter-feature distance $\mu m (min)^{-1}$')
    ax.set_title(title)
    pvalue1 = stats.ttest_ind(val1[:,1],val2[:,1]).pvalue
    x1 = 1
    x2 = 2 
    y = 0.01
    h = 0.002
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    
    title = (
        f"Variation of inter-feature amplitude with dataset \'{dataset_names}\',\n and {len(stat_list_len[:,0])} features tracked"
    )
    _, ax = plt.subplots()
    mask1 = stat_list_height[:,0] <= pole_size
    mask2 = stat_list_height[:,0] > pole_size
    val1 = stat_list_height[mask1]
    val2 = stat_list_height[mask2]
    ax.boxplot([val1[:,1],val2[:,1]],  showfliers=False, showmeans=True, meanline=True) 
    ax.set_xticklabels(['Near pole region', 'Near center region'])
    ax.set_ylabel(r'Variation of inter-feature distance $n m (min)^{-1}$')
    ax.set_title(title)
    pvalue1 = stats.ttest_ind(val1[:,1],val2[:,1]).pvalue
    x1 = 1
    x2 = 2 
    y = 0.01
    h = 0.002
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue1:.2e} ", ha='center', va='bottom')
    
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
    # feature_number('WT_no_drug')
    # feature_creation('WT_no_drug')     #"WT_mc2_55/30-03-2015", "all""no_WT"
    # feature_general_properties("all")
    # feature_creation_comparison("WT_drug",'WT_no_drug')
    # feature_displacement("all") #"all"
    # feature_len_height_variation ("WT_mc2_55/30-03-2015")
    # feature_displacement_comparison("no_WT","WT_drug",'WT_no_drug')
    feature_properties_pole('all')

