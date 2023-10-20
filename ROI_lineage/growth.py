#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:34:34 2023

@author: c.soubrier
"""
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from plot_final_lineage_tree import plot_image_one_ROI #plot_image_lineage_tree,plot_lineage_tree


Directory='WT_INH_700min_2014/'#"dataset/"# 'delta_lamA_03-08-2018/2/'

data_set=['delta_lamA_03-08-2018/','delta_LTD6_04-06-2017/',"delta_parB/03-02-2015/","delta_parB/15-11-2014/","delta_parB/18-01-2015/","delta_parB/18-11-2014/","delta_ripA/14-10-2016/","WT_mc2_55/06-10-2015/","WT_mc2_55/30-03-2015/","WT_mc2_55/03-09-2014/",'WT_INH_700min_2014/','WT_CCCP_irrigation_2016/','WT_filamentation_cipro_2015/']


def surf_growth(ROI,ROI_dict,main_dict,masks_list):
    ID_list=ROI_dict[ROI]['Mask IDs']
    number=len(ID_list)
    surf_val=np.zeros(number)
    time_val=np.zeros(number)
    resolution=main_dict[masks_list[ID_list[0],2]]['resolution']
    
    for i in tqdm.trange(number):
        elem=ID_list[i]
        frame=masks_list[elem][2]
        time_val[i]=main_dict[frame]['time']
        surf_val[i]=main_dict[frame]['area'][masks_list[elem][3]-1]*resolution**2
    return surf_val,time_val





def plot_growth_superimpose(ROI_dict,main_dict,masks_list):
    for ROI in list(ROI_dict.keys()):
        if ROI_dict[ROI]['Parent']!='' and len(ROI_dict[ROI]['Mask IDs'])>=5 and main_dict[masks_list[ROI_dict[ROI]['Mask IDs'][-1],2]]['time']-main_dict[masks_list[ROI_dict[ROI]['Mask IDs'][0],2]]['time']>50:
            surf_val,time_val=surf_growth(ROI,ROI_dict,main_dict,masks_list)
            t_0=time_val[0]
            s_0=surf_val[0]
            newtime=time_val#-t_0*np.ones(len(time_val))
            newsurf=surf_val#-s_0*np.ones(len(surf_val))
            plt.plot(newtime,newsurf,label=ROI_dict[ROI]['index'])
    # plt.legend()
    plt.show()

def plot_growth_all_dataset(ROI_dict,main_dict,masks_list):
    for ROI in list(ROI_dict.keys()):
        surf_val,time_val=surf_growth(ROI,ROI_dict,main_dict,masks_list)
        plt.plot(time_val,surf_val)
        plt.title(ROI)
        plt.show()
        plot_image_one_ROI(ROI,ROI_dict,masks_list,main_dict)
        









if __name__ == "__main__":
    dic_name='Main_dictionnary.npz'

    list_name='masks_list.npz'

    ROI_dict='ROI_dict.npz'

    linmat_name='non_trig_Link_matrix.npy'

    boolmatname="Bool_matrix.npy"

    linkmatname='Link_matrix.npy'

    index_list_name='masks_ROI_list.npz'
    colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]
    for direc in data_set:
        index_list=np.load(direc+index_list_name, allow_pickle=True)['arr_0']
        List_of_masks=np.load(direc+list_name, allow_pickle=True)['arr_0']
        main_dict=np.load(direc+dic_name, allow_pickle=True)['arr_0'].item()
        newdic=np.load(direc+ROI_dict, allow_pickle=True)['arr_0'].item()


        plot_growth_superimpose(newdic,main_dict,List_of_masks)

    # plot_image_one_ROI('ROI 8',newdic,List_of_masks,main_dict)
    # plot_lineage_tree(newdic,List_of_masks,main_dict,colormask,Directory)
    # plot_image_lineage_tree(newdic,List_of_masks,main_dict,colormask,index_list,Directory)
    