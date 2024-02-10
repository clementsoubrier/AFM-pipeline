import os
import sys


import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)
from scaled_parameters import get_scaled_parameters
from peaks_troughs.group_by_cell import load_dataset







def extract_stiffness(frame_data, main_dic, masks_list):
    params = get_scaled_parameters(stiffness=True)
    stif_deriv_prec = params["stif_deriv_prec"] 
    stif_normal_prec = params["stif_normal_prec"]
    stif_tangent_prec = params["stif_tangent_prec"]
    
    line = frame_data["line"]
    mask_index = frame_data["mask_index"]
    
    file = masks_list[mask_index][2]
    index = masks_list[mask_index][3]
    resolution = main_dic[file]['resolution']
    
    deriv_precision = max(1, int(stif_deriv_prec/resolution))
    normal_precision = max(1, int(stif_normal_prec/resolution))
    tangential_precision = max(1, int(stif_tangent_prec/resolution))
    
    stiffness_img = np.load(main_dic[file]['adress'])['DMTModulus_fwd']
    mask = main_dic[file]['masks'] == index
    
    stiff_line = compute_stiffness_line(line, mask, stiffness_img, tangential_precision, normal_precision, deriv_precision)
    return stiff_line
    
    
              
      

def compute_stiffness_line(line, mask, stiffness_img, tangential_precision, normal_precision, deriv_precision):
    stiff_line = np.zeros(len(line))
    for i, pos in enumerate(line):
        n_vec, t_vec = local_frame(i, line, deriv_precision)
        shape = mask.shape
        mask_small = area_mask(pos, shape, n_vec, t_vec, tangential_precision, normal_precision)
        fin_mask = mask_small & mask
        if np.any(fin_mask):
            value = np.average(stiffness_img[fin_mask])
        else :
            value = stiffness_img[pos]
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

def stiffness_pnt_stats(datasetnames):
    
    params = get_scaled_parameters(data_set=True, paths_and_names=True)
    
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
        
        
    for dataset in datasets:
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
                    ys = extract_stiffness(frame_data, main_dict, masks_list)
                    peaks = frame_data["peaks"]
                    troughs = frame_data["troughs"]
                    if len(troughs)+len(peaks)>=3:
                        peaks_list.append(ys[peaks])
                        troughs_list.append(ys[troughs])
                
    peaks_list = np.concatenate(peaks_list)      
    troughs_list = np.concatenate(troughs_list) 
    print(peaks_list,troughs_list)
    _, ax = plt.subplots()
    ax.boxplot([peaks_list, troughs_list], showfliers=False)
    ax.set_title(f"Stiffness with dataset {datasetnames}")
    ax.set_ylabel(r"DMT Modulus ($MPa$)")
    ax.set_xticklabels(["Peaks", "Troughs"])
    # pvalue = stats.ttest_ind(peaks_list, troughs_list).pvalue
    # x1 = 1
    # x2 = 2 
    # y = 0.011
    # h=0.001
    # ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    # ax.text((x1+x2)*.5, y+h, f"P={pvalue:.2e} *", ha='center', va='bottom')
    # plt.show()
    
    
    
    
if __name__ == "__main__":
    stiffness_pnt_stats('no_WT')