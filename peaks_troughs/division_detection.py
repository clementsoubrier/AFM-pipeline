import os
import sys

import matplotlib.pyplot as plt
import numpy as np

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
            if dist_arr[boundary] < max_clos_bdy:
                pos = (1-boundary) * daughter['xs'][0] + boundary * daughter['xs'][-1]
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
    div_list_ori = div_list_ori[(div_list_ori>=0.3) & (div_list_ori<=0.8)]
    div_list = div_list[(div_list>=0.3) & (div_list<=0.5)]
    plt.figure()
    title = (
        f"Division position with dataset \'{datasetnames}\',\n and {len(div_list_ori)} individual features"
    )
    plt.hist(div_list_ori,  color="grey", bins = 20)
    plt.title(title)
    plt.xlabel(r' $\leftarrow \;\text{New pole}\;|\;  \text{old pole} \;\rightarrow$')
    
    
    plt.figure()
    title = (
        f"Division position with dataset \'{datasetnames}\',\n and {len(div_list)} individual features"
    )
    plt.hist(div_list,  color="grey", bins = 20)
    plt.title(title)
    plt.xlabel(r' $\leftarrow \;\text{Pole}\;|\;  \text{Center} \;\rightarrow$')
   
    plt.show()
    

    



if __name__ == "__main__":
    division_statistics('all',use_one_daughter=True) #'all'"WT_mc2_55/30-03-2015"
