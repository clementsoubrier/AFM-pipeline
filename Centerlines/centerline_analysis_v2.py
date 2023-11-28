#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:22:13 2023

@author: c.soubrier
"""

'''Parameters'''

#import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from numba import njit, prange
from numba.typed import List
from tqdm import trange
from multiprocessing import Pool
from functools import partial


package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from scaled_parameters import get_scaled_parameters




'''finding the minimal distance between 2 centerline, within a certain window of horizontal variation (in micrometers). size1, size2 physical dimension of a pixel, epsilon,ratio,max_iter are parameters'''

def distancematrix(dataset,datadirec,dicname,ROIdicname,masklistname,min_size,window,epsilon,ratio,comp_ratio,max_iter):
    center_list,height_list,dist_list,size_list=count_and_order_centerline(dataset,datadirec,dicname,ROIdicname,masklistname,min_size)
    print (type(size_list),type(dist_list),type(height_list))
    dist_matrix,inversion_matrix,delta_matrix=Matrix_construction(height_list,dist_list,size_list,window,epsilon,ratio,comp_ratio,max_iter)
            
    return center_list,dist_matrix,inversion_matrix,delta_matrix



@njit(parallel=True)
def Matrix_construction(height_list,dist_list,size_list,window,epsilon,ratio,comp_ratio,max_iter):
    dim_matrix=len(size_list)
    dist_matrix=np.zeros((dim_matrix,dim_matrix))
    inversion_matrix=np.zeros((dim_matrix,dim_matrix),dtype=np.bool_)
    delta_matrix=np.zeros((dim_matrix,dim_matrix),dtype=np.int32)
    for i in prange(dim_matrix):
        heighti,disti,sizei=height_list[i],dist_list[i],size_list[i]
        for j in prange(i+1,dim_matrix):
            heightj,distj,sizej=height_list[j],dist_list[j],size_list[j]
            (delta,res,inverted)=comparison_centerline(heighti,heightj,disti,distj,sizei,sizej,window,epsilon,ratio,comp_ratio,max_iter)
            dist_matrix[i,j]=res
            dist_matrix[j,i]=res
            inversion_matrix[i,j]=inverted
            inversion_matrix[j,i]=inverted
            delta_matrix[i,j]=delta
            (delta,res,inverted)=comparison_centerline(heightj,heighti,distj,disti,sizej,sizei,window,epsilon,ratio,comp_ratio,max_iter)
            delta_matrix[j,i]=delta
    return dist_matrix,inversion_matrix,delta_matrix

def count_and_order_centerline(dataset,datadirec,dicname,ROIdicname,masklistname,min_size):
    center_list=[]
    height_list=[]
    dist_list=[]
    size_list=[]

    func=partial(count_and_order_centerline_one_data,datadirec=datadirec,dicname=dicname,ROIdicname=ROIdicname,masklistname=masklistname,min_size=min_size)
    with Pool(processes=8) as pool:
        for res in pool.imap_unordered(func,dataset):
            center_list.extend(res[0])
            height_list.extend(res[1])
            dist_list.extend(res[2])
            size_list.extend(res[3])
 
    new_height_list=List()
    new_dist_list=List()
    new_size_list=List()
    [new_height_list.append(x) for x in height_list]
    [new_dist_list.append(x) for x in dist_list]
    [new_size_list.append(x) for x in size_list]

    return center_list, new_height_list, new_dist_list, new_size_list

def count_and_order_centerline_one_data(data,datadirec,dicname,ROIdicname,masklistname,min_size):
    center_list=[]
    height_list=[]
    dist_list=[]
    size_list=[]
    dic=np.load(os.path.join(datadirec, data, dicname), allow_pickle=True)['arr_0'].item()
    ROI_dic=np.load(os.path.join(datadirec, data, ROIdicname), allow_pickle=True)['arr_0'].item()
    mask_list=np.load(os.path.join(datadirec, data, masklistname), allow_pickle=True)['arr_0']
    #taking the list of masks for a better tracking, may insert other type of conditionnal event on the masks
    ROI_list=list(ROI_dic.keys())
    for j in trange(len(ROI_list)):
        ROI=ROI_list[j]
        if len(ROI_dic[ROI]['Mask IDs'])>=5:
            for elem in range(len(ROI_dic[ROI]['masks_quality'])):
                if ROI_dic[ROI]['masks_quality'][elem]:
                    maskid=ROI_dic[ROI]['Mask IDs'][elem]
                    fichier,masknumber=mask_list[maskid][2:]
                    size=dic[fichier]['resolution']
                    line=dic[fichier]['centerlines'][masknumber-1]
                    if len(line)*size>min_size:
                        
                        img=np.load(dic[fichier]['adress'])['Height_fwd']
                        line_data=dist_centerline(line,img)
                        
                        if not line_data[2]:
                            center_list.append([data,maskid,ROI])
                            height_list.append(line_data[0])
                            dist_list.append(line_data[1])
                            size_list.append(size)
    return center_list, height_list, dist_list, size_list
    
@njit 
def dist_centerline(center1,im1):
    n1=len(center1)
    dist1=np.zeros(n1)
    height1=np.zeros(n1)
    error=False
    for i in range(n1):
        height1[i]=im1[center1[i,0],center1[i,1]]
        if i>0:
            comp_norm=norm(center1[i]-center1[i-1])
            dist1[i]=dist1[i-1]+comp_norm
            if comp_norm==0:
                error=True
    return(height1,dist1,error)


@njit               #The result is given by (delta,res,inverted): res the minimal distance, inverted if the second centerline has to be flipped, delta the translation of the beginning of centerline 2 to do (after inversion)
def comparison_centerline(height1,height2,dist1,dist2,size1,size2,window,epsilon,ratio,comp_ratio,max_iter):
    
    (phy_height1,phy_height2,pix_len1,pix_len2)=scaling_centerlines(height1,height2,dist1,dist2,size1,size2)
    
    size=min(size1,size2)
    
    if pix_len1>=pix_len2:
        pixel_drift=int(window/size)
        if pixel_drift>pix_len2*ratio:
            pixel_drift=int(pix_len2*ratio)
        pixel_range=np.array([-pixel_drift,pixel_drift+pix_len1-(pix_len2*comp_ratio[0]//comp_ratio[1]+1)])
        return optimal_trans_center(pix_len1,phy_height1,pix_len2,phy_height2,pixel_range,size*epsilon,comp_ratio,max_iter)
    
    else:
        pixel_drift=int(window/size)
        if pixel_drift>pix_len1*ratio:
            pixel_drift=int(pix_len1*ratio)
        pixel_range=np.array([-pixel_drift,pixel_drift+pix_len2-(pix_len1*comp_ratio[0]//comp_ratio[1]+1)])
        (delta,res,inverted)=optimal_trans_center(pix_len2,phy_height2,pix_len1,phy_height1,pixel_range,size*epsilon,comp_ratio,max_iter)
        if not inverted:
            return (-delta,res,inverted)
        else :
            return (-pix_len2+pix_len1+delta,res,inverted) 

    
    
@njit
def scaling_centerlines(height1,height2,dist1,dist2,size1,size2):
    size=min(size1,size2)
    
    pix_len1=int(dist1[-1]*size1/size)
    pix_len2=int(dist2[-1]*size2/size)
    phy_height1=np.zeros(pix_len1)
    phy_height2=np.zeros(pix_len2)
    

    j=0
    for  i in range(pix_len1):
        if i*size/size1>=dist1[j+1]:
            j+=1
        phy_height1[i]=((dist1[j+1]-i*size/size1)*height1[j+1]+(i*size/size1-dist1[j])*height1[j])/(dist1[j+1]-dist1[j])   
        i+=1

    j=0
    for  i in range(pix_len2):
        if i*size/size2>=dist2[j+1]:
            j+=1
        phy_height2[i]=((dist2[j+1]-i*size/size2)*height2[j+1]+(i*size/size2-dist2[j])*height2[j])/(dist2[j+1]-dist2[j])   
        i+=1
    return (phy_height1,phy_height2,pix_len1,pix_len2)

@njit
def norm(line):
    res=0
    for i in range(len(line)):
        res+=line[i]**2
    return res**(1/2)





@njit   #first element is the longest return the drift, the result, if the second line has to be inverted
def optimal_trans_center(n1,fun1,n2,fun2,pixel_range,epsilon,comp_ratio,max_iter,signed=False):
    if not signed:
        (delta_plus,res_plus,_)=optimal_trans_center(n1,fun1,n2,fun2,pixel_range,epsilon,comp_ratio,max_iter,signed=True)
        (delta_minus,res_minus,_)=optimal_trans_center(n1,fun1,n2,fun2[::-1],pixel_range,epsilon,comp_ratio,max_iter,signed=True)
        if res_plus<=res_minus:
            return (delta_plus,res_plus,False)
        else :
            return (delta_minus,res_minus,True)
    else :
        if max_iter is None or pixel_range[1]-pixel_range[0]<max_iter:
            sub_range= np.linspace(pixel_range[0],pixel_range[1],pixel_range[1]-pixel_range[0]+1)
        
        else:
            sub_range= np.linspace(pixel_range[0],pixel_range[1],max_iter.astype(np.int16))
            
        
        delta=pixel_range[0]
        res=L2_score(n1,fun1,n2,fun2,pixel_range[0],comp_ratio,epsilon)
        
        for i in sub_range:
            score=L2_score(n1,fun1,n2,fun2,int(i),comp_ratio,epsilon)
            # if i>=0:
            #     score=L2_score(n1,fun1,n2,fun2,int(i),epsilon)
            # else :
            #     score=L2_score(n1,fun1,n2,fun2,int(i)-1,epsilon)
            if score<res:
                delta=i
                res=score
            
        if max_iter is None or pixel_range[1]-pixel_range[0]<max_iter:
            return (delta,res,True)

        else:
            width=(pixel_range[1]-pixel_range[0])//max_iter+1
            new_range=np.array([delta-width,delta+width])
            return optimal_trans_center(n1,fun1,n2,fun2,new_range,epsilon,max_iter,signed=True)
        
        







@njit   #first element is the longest

def L2_score(n1, fun1, n2, fun2, delta, comp_ratio, epsilon):
        newfun2=fun2[(comp_ratio[1]-comp_ratio[0])//2*n2//comp_ratio[1]:(comp_ratio[1]+comp_ratio[0])//2*n2//comp_ratio[1]]
        newn2=len(newfun2)
        if delta<0:
            domain=newn2+delta
            func=fun1[:domain]-newfun2[-delta:]
            av=np.average(func)*np.ones(domain)
            return norm(func-av)**2/domain+epsilon*(-delta)
        elif delta>n1-newn2:
            domain=n1-delta
            func=fun1[delta:]-newfun2[:domain]
            av=np.average(func)*np.ones(domain)
            return norm(func-av)**2/domain+epsilon*(delta-(n1-newn2))
        else: 
            domain=newn2
            func=fun1[delta:delta+domain]-newfun2
            av=np.average(func)*np.ones(domain)
            return norm(func-av)**2/domain


def run_centerline_analysis(dataset):
    
    params = get_scaled_parameters(paths_and_names=True,mds=True)
    dic_name = params["main_dict_name"]
    mask_list_name = params["masks_list_name"]
    ROI_dic_name = params["roi_dict_name"]
    data_direc = params["main_data_direc"]
    dir_cent = params["dir_res_centerlines"] 

    epsilon_penal = params['translation_penalty']
    cross_ratio = params['relative_translation_ratio']
    comp_ratio = params['comparision_ratio']
    if comp_ratio[0]>comp_ratio[1]:
        comp_ratio=comp_ratio[::-1]
    if comp_ratio[0]%2==1 or comp_ratio[1]%2==1:
        comp_ratio= 2*comp_ratio
    min_centerline_len = params['min_centerline_len']
    comparision_window = params['mds_max_trans']
    max_iter_opti = params['mds_max_iter']
    

    if not os.path.exists(dir_cent):
        os.makedirs(dir_cent)
        

    (A,B,C,D)=distancematrix(dataset,
                             data_direc,
                             dic_name,
                             ROI_dic_name,
                             mask_list_name,
                             min_centerline_len,
                             comparision_window,
                             epsilon_penal,
                             cross_ratio,
                             comp_ratio,
                             max_iter_opti)

    
    np.save(os.path.join(dir_cent, params['centerline_list']),A)
    np.save(os.path.join(dir_cent, params['distance_matrix']),B)
    np.save(os.path.join(dir_cent, params['inversion_matrix']),C)
    np.save(os.path.join(dir_cent, params['delta_matrix']),D)


def main(Directory= "all"):
    params = get_scaled_parameters(data_set=True)
    if Directory in params.keys():
        run_centerline_analysis(params[Directory])
    elif isinstance(Directory, list)  : 
        run_centerline_analysis(Directory)
    elif isinstance(Directory, str)  : 
        raise NameError('This directory does not exist')
    
    

    



if __name__ == "__main__":
    
    main()
    
  
    
    
    