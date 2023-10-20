#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:08:29 2023

@author: c.soubrier
"""
import cv2
import os
import numpy as np

from plot_final_lineage_tree import plot_image_lineage_tree




Directory="WT_mc2_55/30-03-2015/" #"dataset/"# 'delta_lamA_03-08-2018/2/'

data_set=['delta_lamA_03-08-2018/','delta_LTD6_04-06-2017/',"delta_parB/03-02-2015/","delta_parB/15-11-2014/","delta_parB/18-01-2015/","delta_parB/18-11-2014/","delta_ripA/14-10-2016/","WT_mc2_55/06-10-2015/","WT_mc2_55/30-03-2015/","WT_mc2_55/03-09-2014/",'WT_INH_700min_2014/','WT_CCCP_irrigation_2016/','WT_filamentation_cipro_2015/']




def create_video(direc):

    dicname='Main_dictionnary.npz'

    indexlistname='masks_ROI_list.npz'

    data_direc='data/datasets/'

    main_dict=np.load(data_direc+direc+dicname, allow_pickle=True)['arr_0'].item()
    
    colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]
    
    indexlist=np.load(data_direc+direc+indexlistname, allow_pickle=True)['arr_0']
    
    image_folder = '../Python_code/img/'+direc
    
    video_name = direc[:-1]+'_video.avi'

    plot_image_lineage_tree(main_dict,colormask,indexlist,direc,data_direc,saving=True,img_dir=image_folder)
    

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0,2, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    #cv2.destroyAllWindows()
    video.release()



if __name__ == "__main__":
    create_video(Directory)