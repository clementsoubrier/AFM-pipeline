#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:22:13 2023

@author: c.soubrier
"""

'''Parameters'''

#import matplotlib.pyplot as plt
import os
import numpy as np
from numba import njit
import cv2
from numba.typed import List
from tqdm import trange

data_set=['delta_lamA_03-08-2018/','delta_LTD6_04-06-2017/',"delta_parB/03-02-2015/","delta_parB/15-11-2014/","delta_parB/18-01-2015/","delta_parB/18-11-2014/","delta_ripA/14-10-2016/","WT_mc2_55/06-10-2015/","WT_mc2_55/30-03-2015/","WT_mc2_55/03-09-2014/",'WT_INH_700min_2014/','WT_CCCP_irrigation_2016/','WT_filamentation_cipro_2015/']
# data_set=["WT_mc2_55/03-09-2014/","WT_mc2_55/06-10-2015/","WT_mc2_55/30-03-2015/",'WT_INH_700min_2014/','WT_CCCP_irrigation_2016/','WT_filamentation_cipro_2015/']#

colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]



dic_name='Main_dictionnary.npz'

dim_name='Dimension.npy'

ROI_dic_name='ROI_dict.npz'

mask_list_name='masks_list.npz'


epsilon_penal=0.1

cross_ratio=0.1

min_centerline_len=1.5 #micro meters

comparision_window=0.5 #micro meters

#maximal number of iteration
max_iter_opti=50


'''finding the minimal distance between 2 centerline, within a certain window of horizontal variation (in micrometers). size1, size2 physical dimension of a pixel, epsilon,ratio,max_iter are parameters'''

def distancematrix(dataset,dicname,ROIdicname,min_size,window,epsilon,ratio,max_iter=None):
    center_list,height_list,dist_list,size_list=count_and_order_centerline(dataset,dicname,ROIdicname,min_size)
    dist_matrix,inversion_matrix,delta_matrix=Matrix_construction(height_list,dist_list,size_list,window,epsilon,ratio,max_iter)
            
    return center_list,dist_matrix,inversion_matrix,delta_matrix



@njit
def Matrix_construction(height_list,dist_list,size_list,window,epsilon,ratio,max_iter):
    dim_matrix=len(size_list)
    dist_matrix=np.zeros((dim_matrix,dim_matrix))
    inversion_matrix=np.zeros((dim_matrix,dim_matrix),dtype=np.bool_)
    delta_matrix=np.zeros((dim_matrix,dim_matrix),dtype=np.int32)
    for i in range(dim_matrix):
        heighti,disti,sizei=height_list[i],dist_list[i],size_list[i]
        for j in range(i+1,dim_matrix):
            heightj,distj,sizej=height_list[j],dist_list[j],size_list[j]
            (delta,res,inverted)=comparison_centerline(heighti,heightj,disti,distj,sizei,sizej,window,epsilon,ratio,max_iter=max_iter)
            dist_matrix[i,j]=res
            dist_matrix[j,i]=res
            inversion_matrix[i,j]=inverted
            inversion_matrix[j,i]=inverted
            delta_matrix[i,j]=delta
            (delta,res,inverted)=comparison_centerline(heightj,heighti,distj,disti,sizej,sizei,window,epsilon,ratio,max_iter=max_iter)
            delta_matrix[j,i]=delta
    return dist_matrix,inversion_matrix,delta_matrix

def count_and_order_centerline(dataset,dicname,ROIdicname,min_size):
    center_list=[]
    height_list=List()
    dist_list=List()
    size_list=List()
    count=0
    for i in range(len(dataset)):
        data=dataset[i]
        print(data)
        dic=np.load(data+dicname, allow_pickle=True)['arr_0'].item()
        ROI_dic=np.load(data+ROIdicname, allow_pickle=True)['arr_0'].item()
        mask_list=np.load(data+mask_list_name, allow_pickle=True)['arr_0']
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
                                center_list.append([count,data,maskid,ROI])
                                height_list.append(line_data[0])
                                dist_list.append(line_data[1])
                                size_list.append(size)
                                count+=1
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
def comparison_centerline(height1,height2,dist1,dist2,size1,size2,window,epsilon,ratio,max_iter):
    
    (phy_height1,phy_height2,pix_len1,pix_len2)=scaling_centerlines(height1,height2,dist1,dist2,size1,size2)
    
    size=min(size1,size2)
    
    if pix_len1>=pix_len2:
        pixel_drift=int(window/size)
        if pixel_drift>pix_len2*ratio:
            pixel_drift=int(pix_len2*ratio)
        pixel_range=np.array([-pixel_drift,pixel_drift+pix_len1-(pix_len2*8//10+1)])
        return optimal_trans_center(pix_len1,phy_height1,pix_len2,phy_height2,pixel_range,epsilon,max_iter)
    
    else:
        pixel_drift=int(window/size)
        if pixel_drift>pix_len1*ratio:
            pixel_drift=int(pix_len1*ratio)
        pixel_range=np.array([-pixel_drift,pixel_drift+pix_len2-(pix_len1*8//10+1)])
        (delta,res,inverted)=optimal_trans_center(pix_len2,phy_height2,pix_len1,phy_height1,pixel_range,epsilon,max_iter)
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
def optimal_trans_center(n1,fun1,n2,fun2,pixel_range,epsilon,max_iter,signed=False):
    if not signed:
        (delta_plus,res_plus,plus)=optimal_trans_center(n1,fun1,n2,fun2,pixel_range,epsilon,max_iter,signed=True)
        (delta_minus,res_minus,minus)=optimal_trans_center(n1,fun1,n2,fun2[::-1],pixel_range,epsilon,max_iter,signed=True)
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
        res=L2_score(n1,fun1,n2,fun2,pixel_range[0],epsilon)
        
        for i in sub_range:
            score=L2_score(n1,fun1,n2,fun2,int(i),epsilon)
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

def L2_score(n1,fun1,n2,fun2,delta,epsilon):
        newfun2=fun2[n2//10:9*n2//10]
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

''''''
if __name__ == "__main__":
    
    
    
    # (centerlist,height_list,dist_list,size_list)=count_and_order_centerline(data_set,dic_name,ROI_dic_name,min_centerline_len)
    # print(len(centerlist))
    
    
    
    (A,B,C,D)=distancematrix(data_set,dic_name,ROI_dic_name,min_centerline_len,comparision_window,epsilon_penal,cross_ratio,max_iter=None)
    np.save('centerline_analysis_result/centerline_list_all',A)
    np.save('centerline_analysis_result/distance_matrix_all',B)
    np.save('centerline_analysis_result/inversion_matrix_all',C)
    np.save('centerline_analysis_result/delta_matrix_all',D)
    
    
    # for data in data_set:
    #     if os.path.exists('centerline_analysis_result/'+data):
    #         for file in os.listdir('centerline_analysis_result/'+data):
    #             os.remove(os.path.join('centerline_analysis_result/'+data, file))
    #     else:
    #         os.makedirs('centerline_analysis_result/'+data)
            
    #     (A,B,C,D)=distancematrix([data],dic_name,ROI_dic_name,min_centerline_len,comparision_window,epsilon_penal,cross_ratio,max_iter=None)
    #     np.save('centerline_analysis_result/'+data+'centerline_list',A)
    #     np.save('centerline_analysis_result/'+data+'distance_matrix',B)
    #     np.save('centerline_analysis_result/'+data+'inversion_matrix',C)
    #     np.save('centerline_analysis_result/'+data+'delta_matrix',D)
    
    
    
    