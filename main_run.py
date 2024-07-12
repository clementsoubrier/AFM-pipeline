#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:18:29 2022

@author: c.soubrier
"""
from multiprocessing import Pool

from scaled_parameters import get_scaled_parameters

import processing 
from ROI_lineage import final_graph
from ROI_lineage import plot_final_lineage_tree
from ROI_lineage import Images_to_video
from peaks_troughs import group_by_cell


def run_one_direc(direc):
    processing.run_one_dataset_logs_only(direc)
    final_graph.Final_lineage_tree(direc)
    plot_final_lineage_tree.run_whole_lineage_tree(direc,False)
    Images_to_video.create_video(direc)
    group_by_cell.compute_dataset(direc)




def main(Directory= "all"):
    params = get_scaled_parameters(data_set=True)
    if Directory in params.keys():
        with Pool(processes=8) as pool:
            for direc in pool.imap_unordered(run_one_direc, params[Directory]):
                pass
    elif isinstance(Directory, list)  : 
        with Pool(processes=8) as pool:
            for direc in pool.imap_unordered(run_one_direc, Directory):
                pass
    elif isinstance(Directory, str)  : 
        raise NameError('This directory does not exist')
    
    
    

if __name__ == "__main__":
    main('WT_no_drug')
    # Directory= "WT_mc2_55/30-03-2015"
    # main(Directory= Directory)
    