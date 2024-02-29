import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from scaled_parameters import get_scaled_parameters
from peaks_troughs.group_by_cell import Orientation, load_dataset, get_peak_troughs_lineage_lists, load_cell
from peaks_troughs.preprocess import evenly_spaced_resample
from peaks_troughs.stiffness_stats import extract_feature




def detect_division(mother, roi_id, roi_dic, dataset, use_one_daughter):
    params = get_scaled_parameters(stats=True)
    max_sup = params["div_max_superposition"]
    max_clos_bdy = params["div_max_dist_from_moth"]
    min_daugter = params['div_min_daugther_size']
    children_old = roi_dic[roi_id]['Children']
    children_list = []
    
    
    for child in children_old:
        elem = load_cell(child, dataset=dataset)
        if len (elem)>0:
            children_list.append(child)
            
        
    if len(children_list) == 2:
        daughter_1 = load_cell(children_list[0], dataset=dataset)[0]
        daughter_2 = load_cell(children_list[1], dataset=dataset)[0]
        if daughter_1['xs'][0] > daughter_2['xs'][0]:
            daughter_1, daughter_2 = daughter_2, daughter_1
        if abs(daughter_2['xs'][0] - daughter_1['xs'][-1])<=max_sup:
            pos = (daughter_2['xs'][0] + daughter_1['xs'][-1])/2
            return np.argmin((mother['xs']-pos)**2)
    
    if use_one_daughter:
        if len(children_list) == 1:
            daughter = load_cell(children_list[0], dataset=dataset)[0]
            dist_arr = np.array([abs(mother['xs'][0]- daughter['xs'][0]), abs(mother['xs'][-1]- daughter['xs'][-1])])
            boundary = np.argmin(dist_arr)
            if dist_arr[boundary] < max_clos_bdy and dist_arr[1 - boundary] > min_daugter:
                pos = boundary * daughter['xs'][0] + (1-boundary) * daughter['xs'][-1]
                return np.argmin((mother['xs']-pos)**2)
    return None
    
    
    
def division_statistics(datasetnames, use_one_daughter=False):
    params = get_scaled_parameters(data_set=True)
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    div_list = []
    div_list_ori= []
    for dataset in datasets:
        params = get_scaled_parameters(paths_and_names=True)
        data_direc = params["main_data_direc"]
        roi_dic_name = params["roi_dict_name"]
        roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()

        for roi_id, mother in load_dataset(dataset, False):
            if len(mother)>1:
                mother = mother[-1]
                div_index = detect_division(mother, roi_id, roi_dic, dataset, use_one_daughter)
                if div_index is not None:
                    orientation = Orientation(mother['orientation'])
                    moth_len = mother['xs'][-1] - mother['xs'][0]
                    x_coord = mother['xs'][div_index] - mother['xs'][0]
                    match orientation:
                        case Orientation.NEW_POLE_OLD_POLE:
                            
                            div_list_ori.append(x_coord/moth_len)
                        case Orientation.OLD_POLE_NEW_POLE:
                            div_list_ori.append((moth_len-x_coord)/moth_len)
                            
                    
                    if x_coord > moth_len * 0.5 :
                        x_coord = moth_len - x_coord
                        
                    div_list.append(x_coord/moth_len)
                    
    div_list_ori = np.array(div_list_ori) 
    div_list = np.array(div_list) 
    title = (
        f"Division position with dataset \'{datasetnames}\',\n and {len(div_list_ori)} individual features"
    )
    _, ax = plt.subplots()
    ax.boxplot([div_list_ori], showfliers=False, showmeans=True, meanline=True)
    ax.set_xticklabels(['Division position'])
    ax.set_ylabel(r' $\leftarrow \;\text{New pole}\;|\;  \text{old pole} \;\rightarrow$')
    ax.set_title(title)
    ax.set_ylim(0.38,0.62)
    pvalue = stats.ttest_1samp(div_list_ori,0.5).pvalue #stats.wilcoxon ttest_1samp .pvalue
    ax.text(1.2, 0.5, f"P={pvalue:.2e} *", ha='center', va='bottom')
    
    
    
    plt.figure()
    title = (
        f"Division position with dataset \'{datasetnames}\',\n and {len(div_list)} individual features"
    )
    plt.hist(div_list,  color="grey", bins = 15)
    plt.title(title)
    plt.xlabel(r' $\leftarrow \;\text{Pole}\;|\;  \text{Center} \;\rightarrow$')
   
    plt.show()
    

def closest_feature_amplitude(ind_feature, feature, frame_data):
    non_feature = 'troughs'
    if feature == 'troughs':
        non_feature = 'peaks'
    
    feature_pos = frame_data[feature][ind_feature]
    
    amplitude = None

    mask_l = frame_data[non_feature]<feature_pos
    mask_r = frame_data[non_feature]>feature_pos
    if np.any(mask_l) and np.any(mask_r):
        left = frame_data[non_feature][np.nonzero(mask_l)[0][-1]]
        right = frame_data[non_feature][np.nonzero(mask_r)[0][0]]
        amplitude = abs ((frame_data['ys'][left]+frame_data['ys'][right])/2 - frame_data['ys'][feature_pos])
    elif np.any(mask_l):
        left = frame_data[non_feature][np.nonzero(mask_l)[0][-1]]
        amplitude = abs (frame_data['ys'][left] - frame_data['ys'][feature_pos])
    elif np.any(mask_r):
        right = frame_data[non_feature][np.nonzero(mask_r)[0][0]]
        amplitude = abs (frame_data['ys'][right] - frame_data['ys'][feature_pos])

    return amplitude
    
    
    
    
def division_pnt(datasetnames, use_one_daughter=False):
    params = get_scaled_parameters(data_set=True)
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    pnt_dist_peak = []
    pnt_dist_trough = []
    pnt_height_peak = []
    pnt_height_trough = []
    for dataset in datasets:
        params = get_scaled_parameters(paths_and_names=True)
        data_direc = params["main_data_direc"]
        roi_dic_name = params["roi_dict_name"]
        roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()

        for roi_id, mother in load_dataset(dataset, False):
            if len(mother)>1:
                mother = mother[-1]
                div_index = detect_division(mother, roi_id, roi_dic, dataset, use_one_daughter)
                if div_index is not None:
                    moth_len = mother['xs'][-1] - mother['xs'][0]
                    x_coord = mother['xs'][div_index] - mother['xs'][0]
                    if x_coord > moth_len * 0.5 :
                        x_coord = moth_len - x_coord
                        
                    if len (mother['peaks']) >= 1:
                        closest_peak_ind = np.argmin((mother['xs'][mother['peaks']] - mother['xs'][div_index]) )
                        closest_peak_pos = mother['peaks'][closest_peak_ind]
                        pnt_dist_peak.append(abs  (mother['xs'][closest_peak_pos]- mother['xs'][div_index]))
                    if len (mother['troughs']) >= 1:
                        closest_trough_ind = np.argmin((mother['xs'][mother['troughs']] - mother['xs'][div_index]) )
                        closest_trough_pos = mother['troughs'][closest_trough_ind]
                        pnt_dist_trough.append(abs  (mother['xs'][closest_trough_pos]- mother['xs'][div_index]))

                        
                    if len (mother['troughs']) >= 1 and len (mother['peaks']) >= 1:
                        feature = 'peaks'
                        feature_ind = closest_peak_ind
                        if abs (mother['xs'][closest_trough_pos]- mother['xs'][div_index]) <= abs (mother['xs'][closest_peak_pos]- mother['xs'][div_index]):
                            feature = 'troughs'
                            feature_ind = closest_trough_ind
                        amplitude = closest_feature_amplitude(feature_ind, feature, mother)
                        if amplitude is not None:
                            if feature == 'peaks':
                                pnt_height_peak.append(amplitude)
                            else:
                                pnt_height_trough.append(amplitude)
                    

                    
                    
                    
    pnt_dist_peak = np.array(pnt_dist_peak) 
    pnt_dist_trough = np.array(pnt_dist_trough) 
    pnt_height_peak = np.array(pnt_height_peak) 
    pnt_height_trough = np.array(pnt_height_trough) 
    
    title = (
        f"Division distance to feature with dataset \'{datasetnames}\',\n and {len(pnt_dist_peak)+len(pnt_dist_trough)} individual features"
    )
    _, ax = plt.subplots()
    ax.boxplot([pnt_dist_peak, pnt_dist_trough], showfliers=False, showmeans=True, meanline=True)
    ax.set_xticklabels(["Peak", "Trough"])
    ax.set_ylabel(r'Distance $\mu m $')
    ax.set_title(title)
    pvalue = stats.ttest_ind(pnt_dist_peak, pnt_dist_trough).pvalue
    x1 = 1
    x2 = 2 
    y = 5
    h=0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue:.2e}", ha='center', va='bottom')
    
    
    title = (
        f"Closest feature amplitude with dataset \'{datasetnames}\',\n and {len(pnt_height_peak)+len(pnt_height_trough)} individual features"
    )
    _, ax = plt.subplots()
    ax.boxplot([pnt_height_peak, pnt_height_trough], showfliers=False, showmeans=True, meanline=True)
    ax.set_xticklabels(["Peak", "Trough"])
    ax.set_ylabel(r'Amplitude $nm $')
    ax.set_title(title)
    pvalue = stats.ttest_ind(pnt_height_peak, pnt_height_trough).pvalue
    x1 = 1
    x2 = 2 
    y = 750
    h=20
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue:.2e}", ha='center', va='bottom')
    plt.show()


if __name__ == "__main__":
    division_statistics("all", use_one_daughter=True) #'all'"WT_mc2_55/30-03-2015", use_one_daughter=True
    # division_pnt("all", use_one_daughter=True)