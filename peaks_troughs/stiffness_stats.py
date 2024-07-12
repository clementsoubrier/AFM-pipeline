import os
import sys


import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
from multiprocessing import Pool

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)
from scaled_parameters import get_scaled_parameters
from peaks_troughs.group_by_cell import load_dataset
from functools import partial






def extract_feature(frame_data, main_dic, masks_list, feature, averaged):
    
    
    line = frame_data["line"]
    mask_index = frame_data["mask_index"]
    file = masks_list[mask_index][2]
    index = masks_list[mask_index][3]
    resolution = main_dic[file]['resolution']
    feature_img = np.load(main_dic[file]['adress'])[feature]
    if averaged:
        params = get_scaled_parameters(physical_feature=True)
        stif_deriv_prec = params["phys_deriv_prec"] 
        stif_normal_prec = params["phys_normal_prec"]
        stif_tangent_prec = params["phys_tangent_prec"]
        deriv_precision = max(1, int(stif_deriv_prec/resolution))
        normal_precision = max(1, int(stif_normal_prec/resolution))
        tangential_precision = max(1, int(stif_tangent_prec/resolution))
        mask = main_dic[file]['masks'] == index
        stiff_line = compute_feature_line(line, mask, feature_img, tangential_precision, normal_precision, deriv_precision)
    else :
        stiff_line = feature_img[line[:,0],line[:,1]] 
    return stiff_line
    
    
              
      

def compute_feature_line(line, mask, feature_img, tangential_precision, normal_precision, deriv_precision):
    stiff_line = np.zeros(len(line))
    for i, pos in enumerate(line):
        n_vec, t_vec = local_frame(i, line, deriv_precision)
        shape = mask.shape
        mask_small = area_mask(pos, shape, n_vec, t_vec, tangential_precision, normal_precision)
        fin_mask = mask_small & mask
        if np.any(fin_mask):
            value = np.average(feature_img[fin_mask])
        else :
            value = feature_img[pos[0],pos[1]]
        stiff_line[i] = value
    return stiff_line 
        
        
        

def local_frame(i, line, deriv_precision):  
    if i < deriv_precision :
        data = line[:2*deriv_precision+1]
    elif len(line) - deriv_precision -1 <= i:
        data = line[-2*deriv_precision-1:]
    else:
        data = line[i-deriv_precision:i+deriv_precision+1]
    pca = PCA(n_components=2)
    pca.fit(data)
    res = np.ascontiguousarray(pca.components_)
    t_vec = res[0]
    n_vec = res[1]
    return n_vec, t_vec

@njit
def area_mask(pos, shape, n_vec, t_vec, tangential_precision, normal_precision):
    mask = np.zeros(shape, dtype=np.bool_)
    ran = tangential_precision+normal_precision
    for i in range(-ran,ran+1):
        for j in range(-ran,ran+1):
            vec = np.array([i,j],dtype=np.float_)
            if -normal_precision <= np.dot(vec,n_vec) <= normal_precision \
                and  -tangential_precision <= np.dot(vec,t_vec) <= tangential_precision\
                and 0 <= pos[0]+i < shape[0] \
                and 0 <= pos[1]+j < shape[1]:         
                    mask[pos[0]+i,pos[1]+j] = True
    return mask



def phys_feature_one_dataset(dataset, feature, averaged):
    params = get_scaled_parameters(paths_and_names=True)
    peaks_list = []
    troughs_list = []

    dicname = params["main_dict_name"]
    listname = params["masks_list_name"]
    data_direc = params["main_data_direc"]
    
    masks_list = np.load(os.path.join(data_direc, dataset, listname), allow_pickle=True)['arr_0']
    main_dict = np.load(os.path.join(data_direc, dataset, dicname), allow_pickle=True)['arr_0'].item()
    for _, cells in load_dataset(dataset, False):
        if len(cells) > 3:
            for frame_data in cells:
                ys = extract_feature(frame_data, main_dict, masks_list, feature, averaged)
                peaks = frame_data["peaks"]
                troughs = frame_data["troughs"]
                if len(troughs)+len(peaks)>=3:
                    peaks_list.append(ys[peaks])
                    troughs_list.append(ys[troughs])
    return peaks_list, troughs_list
        
        
        
        
def feature_pnt_stats(datasetnames, feature='DMTModulus_fwd', averaged=True):
    
    params = get_scaled_parameters(data_set=True, paths_and_names=True)
    
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    
    dicname = params["main_dict_name"] 
    data_direc = params["main_data_direc"]
    main_dict = np.load(os.path.join(data_direc, datasets[0], dicname), allow_pickle=True)['arr_0'].item()   
    unit = main_dict[next(iter(main_dict))]['units'][feature]
    
    peaks_list = []
    troughs_list = []
     
    func=partial(phys_feature_one_dataset, feature=feature, averaged=averaged)
    with Pool(processes=8) as pool:
        for pl, tl in pool.imap_unordered(func, datasets):
            peaks_list.extend(pl)
            troughs_list.extend(tl)

    peaks_list = np.concatenate(peaks_list)      
    troughs_list = np.concatenate(troughs_list) 
    _, ax = plt.subplots()
    ax.boxplot([peaks_list, troughs_list], showfliers=False)
    ax.set_title(f"{feature} with dataset {datasetnames} and {len(peaks_list)+len(troughs_list)} features")
    ax.set_ylabel(f"{feature} ({unit})")
    ax.set_xticklabels(["Peaks", "Troughs"])

    plt.show()
    



def phys_region_one_dataset(dataset, feature, averaged):
    params = get_scaled_parameters(paths_and_names=True, physical_feature=True, stats=True)
    non_pole_list = []
    pole_list = []
    diff_list = []

    dicname = params["main_dict_name"]
    listname = params["masks_list_name"]
    data_direc = params["main_data_direc"]
    pole_len = params["pole_region_size"]
    
    masks_list = np.load(os.path.join(data_direc, dataset, listname), allow_pickle=True)['arr_0']
    main_dict = np.load(os.path.join(data_direc, dataset, dicname), allow_pickle=True)['arr_0'].item()
    for _, cells in load_dataset(dataset, False):
        if len(cells) > 3:
            for frame_data in cells:
                ys = extract_feature(frame_data, main_dict, masks_list, feature, averaged)
                xs = frame_data["xs"] - frame_data["xs"][0]
                if xs[-1] >= 2.5*pole_len:
                    mask = (xs>=pole_len) & (xs<=xs[-1]-pole_len)
                    args = np.nonzero(mask)[0]
                    bdr1, bdr2 = args[0], args[-1]
                    
                    pole_list.append(np.average(ys[:bdr1]))
                    pole_list.append(np.average(ys[bdr2:]))
                    diff_list.append((np.average(ys[bdr1:bdr2])-np.average(np.concatenate((ys[:bdr1],ys[bdr2:])))))
                    non_pole_list.append(np.average(ys[bdr1:bdr2]))
                    
    return pole_list, non_pole_list, diff_list
    

def feature_region_stats(datasetnames, feature='DMTModulus_fwd', averaged=True):
    
    params = get_scaled_parameters(paths_and_names=True, data_set=True)
    
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
        
    dicname = params["main_dict_name"] 
    data_direc = params["main_data_direc"]
    main_dict = np.load(os.path.join(data_direc, datasets[0], dicname), allow_pickle=True)['arr_0'].item()   
    unit = main_dict[next(iter(main_dict))]['units'][feature]
    
    pole_list = []
    non_pole_list = [] 
    diff_list = []
    func=partial(phys_region_one_dataset, feature=feature, averaged=averaged)
    with Pool(processes=8) as pool:
        for pl, nl, dl in pool.imap_unordered(func, datasets):
            pole_list.extend(pl)
            non_pole_list.extend(nl)
            diff_list.extend(dl)
    
    pole_list = np.array(pole_list)      
    non_pole_list = np.array(non_pole_list) 
    diff_list = np.array(diff_list)
    
    _, ax = plt.subplots()
    ax.boxplot([pole_list, non_pole_list, diff_list], showfliers=False)
    ax.set_title(f"Region-averaged {feature} with dataset {datasetnames} \n and {len(non_pole_list)} cells")
    ax.set_ylabel(f"{feature} ({unit})")
    ax.set_xticklabels(["Pole", "Non Pole", "Difference Non Pole / Pole"])
    # pvalue = stats.ttest_ind(pole_list,non_pole_list).pvalue
    
    # x1 = 1
    # x2 = 2 
    # y = 0.011
    # h=0.001
    # ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    # ax.text((x1+x2)*.5, y+h, f"P={pvalue:.2e} *", ha='center', va='bottom')

    plt.show()
    
if __name__ == "__main__":
    # feature_pnt_stats("WT_mc2_55/06-10-2015") #'data_with_feature'
    feature_region_stats('WT_no_drug', averaged=False) # 'data_with_feature'