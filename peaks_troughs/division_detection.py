import os
import sys

import matplotlib.pyplot as plt
import numpy as np
# from numpy.polynomial.polynomial.Polynomial import fit
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)
from scipy.ndimage import gaussian_filter1d

from scaled_parameters import get_scaled_parameters
from peaks_troughs.growth_stats import p_value_to_str, extract_growth, print_stats
from peaks_troughs.group_by_cell import Orientation, load_dataset, load_cell, load_data
from peaks_troughs.stats import feature_general_properties




def detect_division(mother, roi_id, roi_dic, dataset, use_one_daughter):
    """_summary_

    Args:
        mother (_type_): _description_
        roi_id (_type_): _description_
        roi_dic (_type_): _description_
        dataset (_TYPE_): _description_
        use_one_daughter (Bool): Using division data with only 1 daughter cell

    Returns:
        int or None: centerline index of division site
    """
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
    """Computing relative division position

    Args:
        datasetnames (str): dataset name
        use_one_daughter (bool, optional): Using division data with only 1 daughter cell. Defaults to False.

    Raises:
        NameError: wrong directory
    """
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
    ax.boxplot([div_list_ori], showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([div_list_ori])
    ax.set_xticklabels(['Division position'])
    ax.set_ylabel(r' $\leftarrow \;\text{New pole}\;|\;  \text{old pole} \;\rightarrow$')
    ax.set_title(title)
    # ax.set_ylim(0.38,0.62)
    pvalue = stats.ttest_1samp(div_list_ori,0.5).pvalue #stats.wilcoxon ttest_1samp .pvalue
    ax.text(1.2, 0.5, p_value_to_str(pvalue), ha='center', va='bottom')
    
    
    
    plt.figure()
    title = (
        f"Division position with dataset \'{datasetnames}\',\n and {len(div_list)} individual features"
    )
    plt.hist(div_list,  color="grey", bins = 15)
    plt.title(title)
    plt.xlabel(r' $\leftarrow \;\text{Pole}\;|\;  \text{Center} \;\rightarrow$')
   
    plt.show()
    


def division_statistics_INH_after_700(use_one_daughter=False):
    """Computing relative division position

    Args:
        datasetnames (str): dataset name
        use_one_daughter (bool, optional): Using division data with only 1 daughter cell. Defaults to False.

    Raises:
        NameError: wrong directory
    """
    
    dataset = 'WT_INH_700min_2014'
    
    div_list_ori= []
    
    params = get_scaled_parameters(paths_and_names=True)
    data_direc = params["main_data_direc"]
    roi_dic_name = params["roi_dict_name"]
    roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()

    for roi_id, mother_ROI in load_dataset(dataset, False):
        if len(mother_ROI)>1:
            mother = mother_ROI[-1]
            if mother['timestamp']>= 700:
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
                        case _:
                            first_pole = np.array([elem['xs'][0] for elem in mother_ROI])
                            second_pole = np.array([elem['xs'][-1] for elem in mother_ROI])
                            time = np.array([elem['timestamp'] for elem in mother_ROI])
                            mat = np.zeros((len(time), 2))
                            mat[:, 0] = time
                            mat[:, 1] = 1
                            sl1, _ = np.linalg.lstsq(mat, first_pole, None)[0]
                            sl2, _ = np.linalg.lstsq(mat, second_pole, None)[0]
                            
                            if sl1 >= 1.2 * sl2:
                                div_list_ori.append((moth_len-x_coord)/moth_len)
                            elif sl2 >= 1.2 * sl1:
                                div_list_ori.append(x_coord/moth_len)
                     
                    
    div_list_ori = np.array(div_list_ori) 
    title = (
        f"Division position with dataset \' INH_after_700\',\n and {len(div_list_ori)} individual features"
    )
    _, ax = plt.subplots()
    ax.boxplot([div_list_ori], showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([div_list_ori])
    ax.set_xticklabels(['Division position'])
    ax.set_ylabel(r' $\leftarrow \;\text{New pole}\;|\;  \text{old pole} \;\rightarrow$')
    ax.set_title(title)
    # ax.set_ylim(0.38,0.62)
    pvalue = stats.ttest_1samp(div_list_ori,0.5).pvalue #stats.wilcoxon ttest_1samp .pvalue
    ax.text(1.2, 0.5, p_value_to_str(pvalue), ha='center', va='bottom')
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
    """Computing closest features to division site, and their properties

    Args:
        datasetnames (str): dataset name
        use_one_daughter (bool, optional): Using division data with only 1 daughter cell. Defaults to False.

    Raises:
        NameError: wrong directory
    """
    params = get_scaled_parameters(data_set=True, paths_and_names=True)
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    
    data_direc = params["main_data_direc"]
    roi_dic_name = params["roi_dict_name"]
    pnt_dist_peak = []
    pnt_dist_trough = []
    pnt_height = {'closest':{'peaks':[], 'troughs': []}, 'second_closest' : {'peaks':[], 'troughs': []}}
    
    for dataset in datasets:
        roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()

        for roi_id, mother in load_dataset(dataset, False):
            if len(mother)>1:
                mother = mother[-1]
                div_index = detect_division(mother, roi_id, roi_dic, dataset, use_one_daughter)
                if div_index is not None:
                    if len (mother['troughs']) >= 1 and len (mother['peaks']) >= 1:
                        closest_peak_ind = np.argmin(np.absolute(mother['xs'][mother['peaks']] - mother['xs'][div_index]) )
                        closest_peak_pos = mother['peaks'][closest_peak_ind]
                        pnt_dist_peak.append(abs  (mother['xs'][closest_peak_pos]- mother['xs'][div_index]))
                        
                        closest_trough_ind = np.argmin(np.absolute(mother['xs'][mother['troughs']] - mother['xs'][div_index]) )
                        closest_trough_pos = mother['troughs'][closest_trough_ind]
                        pnt_dist_trough.append(abs  (mother['xs'][closest_trough_pos]- mother['xs'][div_index]))

                        if abs (mother['xs'][closest_trough_pos]- mother['xs'][div_index]) <= abs (mother['xs'][closest_peak_pos]- mother['xs'][div_index]):
                            pnt_height['second_closest']['peaks'].append(closest_feature_amplitude(closest_peak_ind, 'peaks', mother))
                            pnt_height['closest']['troughs'].append(closest_feature_amplitude(closest_trough_ind, 'troughs', mother))
                        else :
                            pnt_height['closest']['peaks'].append(closest_feature_amplitude(closest_peak_ind, 'peaks', mother))
                            pnt_height['second_closest']['troughs'].append(closest_feature_amplitude(closest_trough_ind, 'troughs', mother))
                            

                    

                    
    diff_list, _, _ = feature_general_properties(datasetnames, plot=False)        
                    
    pnt_dist_peak = np.array(pnt_dist_peak) 
    pnt_dist_trough = np.array(pnt_dist_trough) 

    
    title = (
        f"Division distance to feature with dataset \'{datasetnames}\',\n and {len(pnt_dist_peak)+len(pnt_dist_trough)} individual features"
    )
    _, ax = plt.subplots()
    ax.boxplot([pnt_dist_peak, pnt_dist_trough], showfliers=False,medianprops=dict(color='k'))
    print(title)
    print_stats([pnt_dist_peak, pnt_dist_trough])
    ax.set_xticklabels(["Peak", "Trough"])
    ax.set_ylabel(r'Distance $\mu m $')
    ax.set_title(title)
    pvalue = stats.ttest_ind(pnt_dist_peak, pnt_dist_trough).pvalue
    x1 = 1
    x2 = 2 
    y = 2.7
    h=0.05
    ax.plot([x1,  x2], [y,  y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue), ha='center', va='bottom')
    ax.set_ylim(-0.2,3.2)
    
    
    title = (
        f"Closest feature amplitude with dataset \'{datasetnames}\',\n and {len(pnt_height['closest']['peaks'])+len(pnt_height['closest']['troughs'])} individual features"
    )
    _, ax = plt.subplots()
    ax.boxplot([pnt_height['closest']['peaks']+pnt_height['second_closest']['peaks'],
                pnt_height['closest']['troughs']+pnt_height['second_closest']['troughs'],
                diff_list],
               showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([pnt_height['closest']['peaks']+pnt_height['second_closest']['peaks'],
                pnt_height['closest']['troughs']+pnt_height['second_closest']['troughs'],
                diff_list])
    ax.set_xticklabels(["Peak", "Trough", 'whole features'])
    ax.set_ylabel(r'Amplitude $nm $')
    ax.set_title(title)
    pvalue = stats.ttest_ind(pnt_height['closest']['peaks']+pnt_height['second_closest']['peaks'], pnt_height['closest']['troughs']+pnt_height['second_closest']['troughs']).pvalue
    x1 = 1
    x2 = 2 
    y = 370
    h=0
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue), ha='center', va='bottom')
    pvalue = stats.ttest_ind(pnt_height['closest']['peaks']+pnt_height['second_closest']['peaks'],diff_list).pvalue
    x1 = 1
    x2 = 3 
    y = 410
    h=0
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue), ha='center', va='bottom')
    pvalue = stats.ttest_ind(diff_list, pnt_height['closest']['troughs']+pnt_height['second_closest']['troughs']).pvalue
    x1 = 2
    x2 = 3 
    y = 450
    h=0
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue), ha='center', va='bottom')
    ax.set_ylim(-50,500)
    
    
    print('Comparing features')
    print_stats([pnt_height['closest']['peaks']+pnt_height['second_closest']['peaks']+
                pnt_height['closest']['troughs']+pnt_height['second_closest']['troughs'],
                diff_list])
    print(stats.ttest_ind(diff_list, pnt_height['closest']['peaks']+pnt_height['second_closest']['peaks']+pnt_height['closest']['troughs']+pnt_height['second_closest']['troughs']).pvalue)
    # plt.show()



def division_local_curvature(datasetnames, use_one_daughter=False, smoothing=True):
    """Computing local curvature of division point

    Args:
        datasetnames (str): dataset name
        use_one_daughter (bool, optional): Using division data with only 1 daughter cell. Defaults to False.

    Raises:
        NameError: wrong directory
    """
    
    stat_list = []
    stat_mid = []
    stat_mid2 = []
    stat_old = []
    
    params = get_scaled_parameters(data_set=True, paths_and_names=True,stats=True)
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    
    data_direc = params["main_data_direc"]
    roi_dic_name = params["roi_dict_name"]
    window_phys_size = params['div_curv_phy_window']
    smooth_std = params['div_curv_smooth']
        
    for dataset in datasets:
        roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()
        img_dict, _, _ = load_data(dataset, False)
        frame = list(img_dict.keys())[0]
        pixel_size = img_dict[frame]["resolution"]
        params = get_scaled_parameters(pixel_size=pixel_size, pnt_preprocessing=True, pnt_aligning=True)

        for roi_id, lineage in load_dataset(dataset, False):
            if len(lineage)>1:
                mother = lineage[-1]
                
                    
                div_index = detect_division(mother, roi_id, roi_dic, dataset, use_one_daughter)
                if div_index is not None :
                    
                    mask_window = (mother['xs'] - mother['xs'][div_index])**2 <= window_phys_size**2
                    # sq_delt = (mother['xs'][1]-mother['xs'][0])**2
                    # plt.figure ()
                    # plt.plot(mother['xs'],mother['ys'])
                    # plt.plot(mother['xs'],gaussian_filter1d(mother['ys'], smooth_std/pixel_size, mode="nearest"))
                    # plt.scatter(mother['xs'][div_index],mother['ys'][div_index])
                    
                    if  np.sum(mask_window) > 3 and div_index != 0 and div_index != len(mother['xs'])-1 :
                        if smoothing:
                            window = gaussian_filter1d(mother['ys'], smooth_std/pixel_size, mode="nearest")[mask_window]
                            # stat_list.append((smoothed_line[div_index-1]+smoothed_line[div_index+1]-2*smoothed_line[div_index])/(1000*sq_delt))
                        else:
                            window = mother['ys'][mask_window]
                        x_val = mother['xs'][mask_window]
                        p = np.polyfit(x_val, window, 2)
                        stat_list.append(p[0]/1000)
                    mother_time = mother['timestamp']
                    
                    
                    times = np.array([elem['timestamp'] for elem in lineage])
                    mask= times<=mother_time-90
                    
                    
                    if np.any(mask):
                        mid_mother_ind= np.argmax(times[mask])
                        
                        mid_mother = lineage[mid_mother_ind]
                        index = np.argmin((mid_mother['xs'] - mother['xs'][div_index])**2)
                        
                        mask_window_mid = (mid_mother['xs'] - mother['xs'][div_index])**2 <= window_phys_size**2
                        
                        
                        if  np.sum(mask_window_mid) > 3 and index != 0 and index != len(mid_mother['xs'])-1 :
                            if smoothing:
                                window = gaussian_filter1d(mid_mother['ys'], smooth_std/pixel_size, mode="nearest")[mask_window_mid]
                                # stat_mid.append((smoothed_line[index-1]+smoothed_line[index+1]-2*smoothed_line[index])/(1000*sq_delt))
                            else:
                                window = mid_mother['ys'][mask_window_mid]
                            x_val = mid_mother['xs'][mask_window_mid]
                            
                            p = np.polyfit(x_val, window, 2)
                            stat_mid.append(p[0]/1000)
                    
                    times = np.array([elem['timestamp'] for elem in lineage])
                    mask= times<=mother_time-150
                    
                    if np.any(mask):
                        mid_mother_ind2= np.argmax(times[mask])
                        mid_mother2 = lineage[mid_mother_ind2]
                        index = np.argmin((mid_mother2['xs'] - mother['xs'][div_index])**2)
                        mask_window_mid2 = (mid_mother2['xs'] - mother['xs'][div_index])**2 <= window_phys_size**2
                        
                        
                        if  np.sum(mask_window_mid2) > 3 and index != 0 and index != len(mid_mother2['xs'])-1 :
                            if smoothing:
                                window = gaussian_filter1d(mid_mother2['ys'], smooth_std/pixel_size, mode="nearest")[mask_window_mid2]
                                # stat_mid2.append((smoothed_line[index-1]+smoothed_line[index+1]-2*smoothed_line[index])/(1000*sq_delt))
                            else:
                                window = mid_mother2['ys'][mask_window_mid2]
                            x_val = mid_mother2['xs'][mask_window_mid2]
                            
                            p = np.polyfit(x_val, window, 2)
                            stat_mid2.append(p[0]/1000)
                    
                    old_mother = lineage[0]
                    
                    if mother_time >= old_mother['timestamp'] + 180 :
                        mask_window = (old_mother['xs'] - mother['xs'][div_index])**2 <= window_phys_size**2
                        index = np.argmin((old_mother['xs'] - mother['xs'][div_index])**2)
                        
                        if  np.sum(mask_window) > 3 and index != 0 and index != len(old_mother['xs'])-1 :
                            if smoothing:
                                window = gaussian_filter1d(old_mother['ys'], smooth_std/pixel_size, mode="nearest")[mask_window]
                                # stat_old.append((smoothed_line[index-1]+smoothed_line[index+1]-2*smoothed_line[index])/(1000*sq_delt))
                            else:
                                window = old_mother['ys'][mask_window]
                            x_val = old_mother['xs'][mask_window]
                            
                            p = np.polyfit(x_val, window, 2)
                            stat_old.append(p[0]/1000)
                        
                    
                
    title = (
        f"Curvature value with dataset \'{datasetnames}\' \n and {len(stat_list), len(stat_mid), len(stat_mid2), len(stat_old)} divisions"
    )
    _, ax = plt.subplots()
    ax.boxplot([stat_list, stat_mid, stat_mid2, stat_old], showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([stat_list, stat_mid, stat_mid2, stat_old])
    ax.set_title(title)
    ax.set_xticklabels(["At division", "90 mn before", "150 mn before", 'Previous division'])
    ax.set_ylabel(r'Curvature at division site $\mu m^{-1} $')
    
    pvalue = stats.wilcoxon(stat_list, method='exact').pvalue  #stats.wilcoxon
    ax.text(1.3, 0.3, p_value_to_str(pvalue), ha='center', va='bottom')
    
    pvalue = stats.wilcoxon(stat_mid, method='exact').pvalue 
    ax.text(2.2, 0.3, p_value_to_str(pvalue), ha='center', va='bottom')
    
    pvalue = stats.wilcoxon(stat_mid2, method='exact').pvalue 
    ax.text(3.2, 0.3, p_value_to_str(pvalue), ha='center', va='bottom')
    
    pvalue = stats.wilcoxon(stat_old, method='exact').pvalue 
    ax.text(4.2, 0.3, p_value_to_str(pvalue), ha='center', va='bottom')
    ax.set_ylim([-0.05,0.35])
    plt.tight_layout()
    plt.show()



def division_local_curvature_trajectories(datasetnames, use_one_daughter=False, smoothing=True):
    """Computing local curvature over time of division point position

    Args:
        datasetnames (str): dataset name
        use_one_daughter (bool, optional): Using division data with only 1 daughter cell. Defaults to False.

    Raises:
        NameError: wrong directory
    """
    
    stat_list = []
    traj = []
    
    params = get_scaled_parameters(data_set=True, paths_and_names=True,stats=True)
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    
    data_direc = params["main_data_direc"]
    roi_dic_name = params["roi_dict_name"]
    window_phys_size = params['div_curv_phy_window']
    smooth_std = params['div_curv_smooth']
    
    for dataset in datasets:
        roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()
        img_dict, _, _ = load_data(dataset, False)
        frame = list(img_dict.keys())[0]
        pixel_size = img_dict[frame]["resolution"]

        for roi_id, lineage in load_dataset(dataset, False):
            if len(lineage)>1:
                local_traj = []
                mother = lineage[-1]
                mother_time = mother['timestamp']
                    
                div_index = detect_division(mother, roi_id, roi_dic, dataset, use_one_daughter)
                if div_index is not None :
                    for daughter in lineage :
                        daughter_time = daughter['timestamp']
                        mask_window = (daughter['xs'] - mother['xs'][div_index])**2 <= window_phys_size**2
                        if  np.sum(mask_window) > 3 :
                            if smoothing:
                                window = gaussian_filter1d(daughter['ys'], smooth_std/pixel_size, mode="nearest")[mask_window]
                            else:
                                window = daughter['ys'][mask_window]
                            x_val = daughter['xs'][mask_window]
                            
                            p = np.polyfit(x_val, window, 2)
                            stat_list.append([daughter_time - mother_time, p[0]/1000])
                            local_traj.append([daughter_time - mother_time, p[0]/1000])
                    traj.append(np.array(local_traj))
    
    stat_list = np.array(stat_list)
    plt.scatter(stat_list[:,0], stat_list[:,1], color = 'k')
    
    
    x_min = -200
    x_max = 0
    p = np.polyfit(stat_list[:,0], stat_list[:,1], 2)
    plt.plot(np.linspace(x_min, x_max, 100), p[2] + p[1]*np.linspace(x_min, x_max, 100) + p[0]*np.linspace(x_min, x_max, 100)**2 , color = 'r')
    
    plt.title('Curvature at division site over time ')
    plt.ylabel(r'Curvature $\mu m^{-1}$')
    plt.xlabel('time before division (mn)')
    
    plt.figure()
    for elem in traj:
        # t_min = np.min(elem[:,0])
        # t_max = np.max(elem[:,0])
        # p = np.polyfit(elem[:,0], elem[:,1], 1)
        # plt.plot(np.linspace(t_min, t_max, 100), p[1] + p[0]*np.linspace(t_min, t_max, 100))
        plt.plot(elem[:,0], elem[:,1])
    plt.show()
                    
                            
              
                    
                    
                    
                    
                    
                    







if __name__ == "__main__":
    plt.rcParams.update({'font.size': 13})
    # division_statistics_INH_after_700(use_one_daughter=True) 
    # division_statistics('WT_INH_700min_2014', use_one_daughter=True)
    # division_statistics("WT_no_drug", use_one_daughter=True)
    division_pnt('WT', use_one_daughter=True)
    # division_pnt("WT_mc2_55/30-03-2015", use_one_daughter=True)
    # division_pnt('WT_no_drug', use_one_daughter=True)
    # division_pnt('all', use_one_daughter=True)
    # division_local_curvature("WT_mc2_55/30-03-2015", use_one_daughter=True, smoothing=True)  #'all' "WT_mc2_55/30-03-2015"'WT_no_drug'
    # division_local_curvature_trajectories("WT_mc2_55/30-03-2015", use_one_daughter=True, smoothing = True)
    plt.rcParams.update({'font.size': 10})
    plt.show()