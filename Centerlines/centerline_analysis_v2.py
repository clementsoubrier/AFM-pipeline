#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:22:13 2023

@author: c.soubrier
"""

'''Parameters'''

#import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import cv2
from numba.typed import List

data_set= ['dataset/']#["delta_3187/21-02-2019/","delta_3187/19-02-2019/","delta parB/03-02-2015/","delta parB/15-11-2014/","delta parB/18-11-2014/","delta_lamA_03-08-2018/1/","delta_lamA_03-08-2018/2/"]  #Anti"WT_mc2_55/30-03-2015/",?"WT_mc2_55/04-06-2016/",Anti"WT_mc2_55/05-02-2014/",Anti"WT_11-02-15/",      #Yes "delta_3187/21-02-2019/","delta_3187/19-02-2019/","delta parB/03-02-2015/","delta parB/15-11-2014/","delta parB/18-11-2014/","delta_lamA_03-08-2018/1/","delta_lamA_03-08-2018/2/"          #Maybe "delta parB/18-01-2015/",                #No "delta ripA/14-10-2016/","delta ripA/160330_rip_A_no_inducer/","delta ripA/160407_ripA_stiffness_septum/","Strep_pneumo_WT_29-06-2017/","Strep_pneumo_WT_07-07-2017/"


colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]

result_path='Height/Dic_dir/'

dic_name='Main_dictionnary.npy'

dim_name='Dimension.npy'

mask_list_name='masks_list.npy'

graph_name='basic'

epsilon_penal=0.1

cross_ratio=0.1

min_centerline_len=1.5 #micro meters

comparision_window=0.5 #micro meters

#maximal number of iteration
max_iter_opti=50


'''finding the minimal distance between 2 centerline, within a certain window of horizontal variation (in micrometers). size1, size2 physical dimension of a pixel, epsilon,ratio,max_iter are parameters'''

def distancematrix(dataset,resultpath,dicname,min_size,window,epsilon,ratio,max_iter=None):
    center_list,height_list,dist_list,size_list=count_and_order_centerline(dataset,resultpath,dicname,min_size)
    dist_matrix,inversion_matrix,delta_matrix=Matrix_construction(height_list,dist_list,size_list,window,epsilon,ratio,max_iter)
            
    return center_list,dist_matrix,inversion_matrix,delta_matrix



@jit
def Matrix_construction(height_list,dist_list,size_list,window,epsilon,ratio,max_iter):
    dim_matrix=len(size_list)
    dist_matrix=np.zeros((dim_matrix,dim_matrix))
    inversion_matrix=np.zeros((dim_matrix,dim_matrix),dtype=np.bool8)
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

def count_and_order_centerline(dataset,resultpath,dicname,min_size):
    center_list=[]
    height_list=List()
    dist_list=List()
    size_list=List()
    count=0
    for data in dataset:
        dic=np.load(data+resultpath+dicname, allow_pickle=True).item()
        mask_list=np.load(data+resultpath+mask_list_name, allow_pickle=True)
        #taking the list of masks for a better tracking, may insert other type of conditionnal event on the masks
        for i in range(len(mask_list)):
            fichier,masknumber=mask_list[i][2:]
            size=dic[fichier]['resolution']
            line=dic[fichier]['centerlines'][masknumber-1]
            if len(line)*size>min_size:
                center_list.append([count,data,i])
                img=cv2.imread(dic[fichier]['adress'],0)
                line_data=dist_centerline(line,img)
                height_list.append(line_data[0])
                dist_list.append(line_data[1])
                size_list.append(size)
                count+=1
    return center_list, height_list, dist_list, size_list


@jit 
def dist_centerline(center1,im1):
    n1=len(center1)
    dist1=np.zeros(n1)
    height1=np.zeros(n1,dtype=np.uint8)
    for i in range(n1):
        height1[i]=im1[center1[i,0],center1[i,1]]
        if i>0:
            dist1[i]=dist1[i-1]+norm(center1[i]-center1[i-1])
    return(height1,dist1)


@jit               #The result is given by (delta,res,inverted): res the minimal distance, inverted if the second centerline has to be flipped, delta the translation of the beginning of centerline 2 to do (after inversion)
def comparison_centerline(height1,height2,dist1,dist2,size1,size2,window,epsilon,ratio,max_iter):
    
    (phy_height1,phy_height2,pix_len1,pix_len2)=scaling_centerlines(height1,height2,dist1,dist2,size1,size2)
    
    size=min(size1,size2)
    
    if pix_len1>=pix_len2:
        pixel_drift=int(window/size)
        if pixel_drift>pix_len2*ratio:
            pixel_drift=int(pix_len2*ratio)
        pixel_range=np.array([-pixel_drift,pixel_drift+pix_len1-pix_len2])
        return optimal_trans_center(pix_len1,phy_height1,pix_len2,phy_height2,pixel_range,epsilon,max_iter)
    
    else:
        pixel_drift=int(window/size)
        if pixel_drift>pix_len1*ratio:
            pixel_drift=int(pix_len1*ratio)
        pixel_range=np.array([-pixel_drift,pixel_drift+pix_len2-pix_len1])
        (delta,res,inverted)=optimal_trans_center(pix_len2,phy_height2,pix_len1,phy_height1,pixel_range,epsilon,max_iter)
        if not inverted:
            return (-delta,res,inverted)
        else :
            return (-pix_len2+pix_len1+delta,res,inverted) 

    
    
@jit
def scaling_centerlines(height1,height2,dist1,dist2,size1,size2):
   
    size=min(size1,size2)
    
    #print(dist1,dist2)
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
    #print(pix_len2)
    for  i in range(pix_len2):
        if i*size/size2>=dist2[j+1]:
            j+=1
        phy_height2[i]=((dist2[j+1]-i*size/size2)*height2[j+1]+(i*size/size2-dist2[j])*height2[j])/(dist2[j+1]-dist2[j])   
        i+=1
    return (phy_height1,phy_height2,pix_len1,pix_len2)

@jit
def norm(line):
    res=0
    for i in range(len(line)):
        res+=line[i]**2
    return res**(1/2)





@jit   #first element is the longest return the drift, the result, if the second line has to be inverted
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
            sub_range= np.linspace(pixel_range[0],pixel_range[1],max_iter)
            
        
        delta=pixel_range[0]
        res=L2_score(n1,fun1,n2,fun2,pixel_range[0],epsilon)
        
        for i in sub_range:
            if i>=0:
                score=L2_score(n1,fun1,n2,fun2,int(i),epsilon)
            else :
                score=L2_score(n1,fun1,n2,fun2,int(i)-1,epsilon)
            if score<res:
                delta=i
                res=score
            
        if max_iter is None or pixel_range[1]-pixel_range[0]<max_iter:
            return (delta,res,True)

        else:
            width=(pixel_range[1]-pixel_range[0])//max_iter+1
            new_range=np.array([delta-width,delta+width])
            return optimal_trans_center(n1,fun1,n2,fun2,new_range,epsilon,max_iter,signed=True)
        
        







@jit   #first element is the longest
def L2_score(n1,fun1,n2,fun2,delta,epsilon):
        if delta<0:
            domain=n2+delta
            func=fun1[:domain]-fun2[-delta:]
            av=np.average(func)*np.ones(domain)
            return norm(func-av)**2/domain+epsilon*(-delta)
        elif delta>n1-n2:
            domain=n1-delta
            func=fun1[delta:]-fun2[:domain]
            av=np.average(func)*np.ones(domain)
            return norm(func-av)**2/domain+epsilon*(delta-(n1-n2))
        else: 
            domain=n2
            func=fun1[delta:delta+domain]-fun2
            av=np.average(func)*np.ones(domain)
            return norm(func-av)**2/domain


    
''''''
if __name__ == "__main__":
    (centerlist,height_list,dist_list,size_list)=count_and_order_centerline(data_set,result_path,dic_name,min_centerline_len)
    print(len(centerlist))
    (A,B,C,D)=distancematrix(data_set,result_path,dic_name,min_centerline_len,comparision_window,epsilon_penal,cross_ratio,max_iter=None)
    np.save('centerline_analysis_result/centerline_list_3',A)
    np.save('centerline_analysis_result/distance_matrix_3',B)
    np.save('centerline_analysis_result/inversion_matrix_3',C)
    np.save('centerline_analysis_result/delta_matrix_3',D)
    
    
    
    