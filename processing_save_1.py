#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:18:29 2022

@author: c.soubrier
"""

#First test AFM

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numba import jit
import tools2
from cellpose import utils, io, models, plot


'''
from scipy.interpolate import splev
from IPython.core.display import HTML
import sys
import pandas as pd
from scipy.spatial import distance as dist
import re
from radfil import radfil_class, styles
from astropy import units as u
import imageio.v2 as imageio
import copy
from PIL import Image
from fil_finder import FilFinder2D
from shapely.geometry import Polygon
from skimage.util import invert
from skimage.morphology import skeletonize
from scipy.interpolate import splprep
import scipy.optimize as opt
from pystackreg import StackReg
'''


#WARNING : this code better works way better if there is no zoom (very important) between the images (cellpose) and work better if all images have the same size 
# to dicuss : pre-treatment of the images to get no zoom and same format



''' Running in a in a reduced dataset? '''
test=False


''' Paths of the data and to save the results'''
#directory of the original dataset composed of a sequence of following pictures of the same bacterias
my_data = (1-test)*"dataset/" +test*"datatest/"
#directory of cropped data each image of the dataset is cropped to erase the white frame
cropped_data=(1-test)*"cropped_data/" +test*"cropped_datatest/"
#directory for output data from cellpose 
segments_path = (1-test)*"cellpose_output/"+test*"cellpose_outputtest/"
#Saving path for the dictionnary
saving_path='Main_dictionnary'


''' Parameters'''
#cropping parameters
crop_up=27
crop_down=1
crop_left=30
crop_right=101

#cellpose parameters
cel_gpu=True
cel_model_type='cyto'
cel_channels=[0,0]  # define CHANNELS to run segementation on grayscale=0, R=1, G=2, B=3; channels = [cytoplasm, nucleus]; nucleus=0 if no nucleus
cel_diameter = 60
cel_flow_threshold = 0.9 
cel_cellprob_threshold=0.0



#erasing small masks that have a smaller relative size :
ratio_erasing_masks=0.1

#number of point of the approximated convex hull (at least 3)
hull_point=4

#fraction of the preserved area to consider child and parent relation for masks
surface_thresh=0.5




''' main dictionnary main_dict: each image contain a dictionnary with
time : the time of the image beginning at 0
adress : location of the cropped file
masks : the masks as numpy array
outlines : the outlines as a list of points
centroid : an array containing the position of the centroid of each mask the centroid[i] is the centroid of the mask caracterized by i+1
area : an array containing the position of the area (number of pixels) of each mask
mask_error : True if cellpose computes no mask, else False
convex_hull : convex hull of each mask
main_centroid : the centroid of all masks
parent / child : previous / next acceptable  file
mask_parent/child/grand_parent/grand_child : tracking of the mask (area changes) over 1 and 2 generations
translation_vector : vector to superimpose parent and child picture
graph_name+'graph' : graph representing the relation between masks
graph_name+'graph_values': value to take for each mask (continuity of the mask)
graph_name+'masks' : mask with updated value
'''



''' Preparation of the data (sorting and cropping). Here data is a file with pictures of the cell (.png)'''
def data_prep(data,cropup,cropdown,cropleft,cropright):
    # Load up a list of input files (name as strings) from our example data. Creating the main dictionnay containing all tha information 
    files = os.listdir(data)
    
    # Sort files by timepoint.
    files.sort(key = tools2.natural_keys)     
    
    # removing the non image files, and creating the keys of the dictionnay
    for i in range(len(files)):
        if not(files[i].endswith(".png")):
            files.remove(files[i])
        
    #clean or create the cropped directory
    if os.path.exists(cropped_data):
        for file in os.listdir(cropped_data):
            os.remove(os.path.join(cropped_data, file))
    else:
        os.makedirs(cropped_data)
    
    #cropping the data to erase the white frame : to do manually depending on the dataset. Saving the cropped images
    for fichier in files:
        img = cv2.imread(my_data+ fichier)
        cropped = img[cropup:-cropdown,cropleft:-cropright]
        cv2.imwrite(cropped_data+fichier, cropped)  



''' Running cellpose and saving the results'''
def run_cel(croppeddata,gpuval,mod,chan,dia,thres,celp,seg):
    #clean or create the output directory
    if os.path.exists(seg):
        for file in os.listdir(seg):
            os.remove(os.path.join(seg, file))
    else:
        os.makedirs(seg)
    # Specify that the cytoplasm Cellpose model for the segmentation. 
    model = models.Cellpose(gpu=gpuval, model_type=mod)
    # Loop over all of our image files and run Cellpose on each of them. 
    for fichier in os.listdir(croppeddata):
        img = io.imread(croppeddata+fichier)
        masks, flows, st, diams = model.eval(img, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)
        # save results so you can load in gui
        io.masks_flows_to_seg(img, masks, flows, diams, seg+fichier[:-4], chan)



''' Creation of a dictionnary with the entry : adress, time, mask, outlines, mask_error'''
def download_dict(croppeddata,segmentspath):
    files = os.listdir(croppeddata)
    dic={}
    # Sort files by timepoint.
    files.sort(key = tools2.natural_keys)
    t=0
    for fichier in files:
        fichier=fichier[:-4]
        dic[fichier]={}
        dat = np.load(segmentspath+fichier+'_seg.npy', allow_pickle=True).item()
        dic[fichier]['time']=t
        t+=1
        dic[fichier]['adress']=croppeddata+fichier+'.png'
        dic[fichier]['masks']=dat['masks']
        dic[fichier]['outlines']=utils.outlines_list(dat['masks'])
        dic[fichier]['masks_error']=(np.max(dic[fichier]['masks'])==0)
    return dic



''' Construction the sequence of usable pictures : linking images with previous (parent) and following (child) image'''
def main_parenting(dic):
    parent=''
    for fichier in dic.keys():
        if not dic[fichier]['masks_error']:
            dic[fichier]['parent']=parent
            parent=fichier
    child=''
    key=list(dic.keys())
    key.reverse()
    for fichier in key:
        if not dic[fichier]['masks_error']:
            dic[fichier]['child']=child
            child=fichier



''' Computing the area and the centroid of each mask and saving them in the dictionnary'''
def centroid_area(dic):
    for fichier in dic.keys():
        if not dic[fichier]['masks_error']:
            masks=dic[fichier]['masks']
            mask_number=np.max(masks)
            centroid=np.zeros((mask_number,2))
            area=np.zeros(mask_number)
            (l,L)=np.shape(masks)
            for i in range(mask_number):
                count=0
                vec1=0
                vec2=0
                for j in range(L):
                    for k in range(l):
                        if masks[k,j]==i+1:
                            vec1+=j
                            vec2+=k
                            count+=1
                area[i]=count
                centroid[i,:]=np.array([vec1//count,vec2//count])
        dic[fichier]['centroid']=centroid
        dic[fichier]['area']=area



''' Erasing too small masks (less than the fraction frac of the largest mask). Creating the centroid of the union of acceptable  mask and saving as main_centroid'''
def clean_masks(fraction,dic,saving=False,savingpath='dict'):
    #Erase the masks that are too small (and the centroids too)
    for fichier in dic.keys():
        masks=dic[fichier]['masks']
        if not dic[fichier]['masks_error']:
            area=dic[fichier]['area']
            centroid=dic[fichier]['centroid']
            max_area=np.max(area)
            L=len(area)
            non_defect=np.zeros(L) #classification of the allowed masks
            non_defect_count=0
            
            for i in range(L):
                if area[i]>=fraction*max_area:
                    non_defect_count+=1
                    non_defect[i]=non_defect_count
            #new value of the area and the centroid
            area2=np.zeros(non_defect_count)
            centroid2=np.zeros((non_defect_count,2))
            for i in range(L):
                if non_defect[i]!=0:
                    area2[int(non_defect[i]-1)]=area[i]
                    centroid2[int(non_defect[i]-1),:]=centroid[i,:]
            (m,n)=masks.shape
            for j in range(m):
                for k in range(n):
                    if masks[j,k]!=0:
                        masks[j,k]=non_defect[masks[j,k]-1]
            
            dic[fichier]['area']=area2
            dic[fichier]['centroid']=centroid2
            #constructing the main centroid
            main_centroid0=0
            main_centroid1=0
            for i in range (non_defect_count):
                main_centroid0+=area2[i]*centroid2[i,0]
                main_centroid1+=area2[i]*centroid2[i,1]
            dic[fichier]['main_centroid']=np.array([main_centroid0//sum(area2),main_centroid1//sum(area2)])
            if saving:
                np.save(savingpath,main_dict)



''' Ploting the images with the masks overlay, ploting the convex hulls if test_convex is True'''
def plot_masks(fichier,dic,test_convex=False):

    # plot image with masks overlaid
    img = cv2.imread(dic[fichier]['adress'],0)
    #plt.imshow(img)
    masks=dic[fichier]['masks']
    mask_RGB = plot.mask_overlay(img,masks)
    plt.imshow(mask_RGB)
    # plot the centroids
    if not dic[fichier]['masks_error']:
        centr=dic[fichier]['centroid']
        for i in range(len(centr)):
            plt.plot(centr[i,0], centr[i,1], color='k',marker='o')
        if 'main_centroid' in dic[fichier].keys():
            main_centroid=dic[fichier]['main_centroid']
            plt.plot(main_centroid[0], main_centroid[1], color='w',marker='o')
        plt.show()
        if test_convex:
            plt.imshow(img,'Greys_r') 
            colo=['b','g','r','c','m','y']
            mask_number=np.max(masks)
            (l,L)=np.shape(masks)
            convex_plot=np.zeros((l,L))
            convex=dic[fichier]['convex_hull']
            for i in range(l):
                for j in range(L):
                    if convex_plot[i,j]==0:
                        ispresent=[]
                        for k in range(mask_number):
                            if convex[k,i,j]!=0:
                                ispresent.append(convex[k,i,j])
                        lenis=len(ispresent)
                        if lenis==1:
                            convex_plot[i,j]=ispresent[0]
                        elif lenis>1:
                            convex_plot[i,j]=ispresent[np.random.randint(lenis)]
            for i in range(l):
                for j in range(L):
                     if convex_plot[i,j]!=0:
                         plt.plot(j, i, color=colo[int(convex_plot[i,j]%6)],marker=',')
            plt.show()
            


''' Creating an approximation of the convex hull of each mask, which is a polygon of pointnumber vertices'''
def convex_hull(dic,pointnumber,saving=False,savingpath='dict'):
    #create an approximated convex hull of each mask (inside a polygone with pointnumber vertices): each hull is in a numpy array and has the same sumber as the mask
    for fichier in dic.keys():
        if not dic[fichier]['masks_error']:
            masks=dic[fichier]['masks']
            outlines=dic[fichier]['outlines']
            mask_number=np.max(masks)
            (l1,l2)=np.shape(masks)
            convex=np.zeros((mask_number,l1,l2))
            for i in range(mask_number):
                stepsize=int(len(outlines[i])//pointnumber)
                convex[i]=convex_numba(i+1,masks,outlines[i][:-1:stepsize])
            plt.show()
            dic[fichier]['convex_hull']=convex
            if saving:
                np.save(savingpath,dic)

# Function to effectively compute the approximation, 
@jit 
def convex_numba(n0,masks,outline):
    (l1,l2)=np.shape(masks)
    convex=np.zeros((l1,l2))
    lout=len(outline)
    for j in range(lout-2):
        for k in range(j+1,lout-1):
            for l in range(k+1,lout):
                for m in range(l1):
                    for p in range(l2):
                        if masks[m,p]==n0:
                            convex[m,p]=n0
                        else:
                            curl1=(p-outline[j,0])*(outline[j,1]-outline[k,1])-(m-outline[j,1])*(outline[j,0]-outline[k,0])
                            curl2=(p-outline[k,0])*(outline[k,1]-outline[l,1])-(m-outline[k,1])*(outline[k,0]-outline[l,0])
                            curl3=(p-outline[l,0])*(outline[l,1]-outline[j,1])-(m-outline[l,1])*(outline[l,0]-outline[j,0])
                            if (curl1>=0 and curl2>=0 and curl3>=0 and (curl1>0 or curl2>0 or curl3>0)) or (curl1<=0 and curl2<=0 and curl3<=0 and (curl1<0 or curl2<0 or curl3<0)):
                                convex[m,p]=n0
    return convex



''' Computing the translation vector between an image and its child and saving it under translation_vector'''
def mask_displacement(dic,rad):
    ''' Older version of the algorithm
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        child=dic[fichier]['child']
        shape_1=np.shape(dic[fichier]['masks'])
        shape_2=np.shape(dic[child]['masks'])
        shape_f=(min(shape_1[0],shape_2[0]),min(shape_1[1],shape_2[1]))
        mask_p=main_mask(dic[fichier]['masks'][:shape_f[0],:shape_f[1]])
        mask_c=main_mask(dic[child]['masks'][:shape_f[0],:shape_f[1]])
        sr = StackReg(StackReg.SCALED_ROTATION)
        sr.register_transform(mask_p,mask_c)
        dic[fichier]['translation_vector']=np.int_(sr.get_matrix())[:2,2]
        fichier=child
    '''
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        child=dic[fichier]['child']
        shape_1=np.shape(dic[fichier]['masks'])
        shape_2=np.shape(dic[child]['masks'])
        shape_f=(min(shape_1[0],shape_2[0]),min(shape_1[1],shape_2[1]))
        mask_p=main_mask(dic[fichier]['masks'][:shape_f[0],:shape_f[1]])
        mask_c=main_mask(dic[child]['masks'][:shape_f[0],:shape_f[1]])
        vecguess=dic[fichier]['main_centroid']-dic[child]['main_centroid']
        dic[fichier]['translation_vector']=opt_trans_vec(mask_p,mask_c,rad,vecguess)
        fichier=child

# Tranform all masks into one shape (the main shape)
@jit
def main_mask(mask):
    (l,L)=np.shape(mask)
    new_mask=np.zeros((l,L))
    for i in range(l):
        for j in range(L):
            if mask[i,j]!=0:
                new_mask[i,j]=1
    return new_mask

# Define the score of a function (here a sum of white pixels)
@jit 
def score_mask(mask1,mask2):
    (l1,l2)=np.shape(mask1)
    score=0
    for i in range(l1):
        for j in range(l2):
            if mask1[i,j]==1 and mask2[i,j]==1:
                score+=1
    return score
         
# Translation of the masks by a vector
@jit 
def mask_transfert(mask,vector):
    (l1,l2)=np.shape(mask)
    new_mask=np.zeros((l1,l2))
    for i in range(l1):
        for j in range(l2):
            if (0<=i+vector[0]<=l1-1) and (0<=j+vector[1]<=l2-1) and mask[i,j]>0:
                new_mask[int(i+vector[0]),int(j+vector[1])]=mask[i,j]
    return new_mask
      
# Effective computation of the translation vector : returns the translation vector that optimizes the score of the intersection of the two main shapes
@jit
def opt_trans_vec(mask1,mask2,rad,vec_guess):
    # first part of the algorithm
    #peut etre faire une PCA pour trouver l'axe principal des cellules, on suppose au'il est vertical/horizontal
    scoring=score_mask(mask1,mask_transfert(mask2, vec_guess))
    arg=vec_guess
    termination=False
    while not termination:
        seg=np.arange(-rad,rad+1,1)
        seg_len=2*rad+1
        res=np.zeros(4*seg_len)
        for i in range(seg_len):
            res[i]=score_mask(mask1,mask_transfert(mask2, vec_guess+np.array([seg[i],0])))
        for i in range(seg_len):
            res[i+seg_len]=score_mask(mask1,mask_transfert(mask2, vec_guess+np.array([0,seg[i]])))
        for i in range(seg_len):
            res[i+2*seg_len]=score_mask(mask1,mask_transfert(mask2, vec_guess+np.array([seg[i],seg[i]])))
        for i in range(seg_len):
            res[i+3*seg_len]=score_mask(mask1,mask_transfert(mask2, vec_guess+np.array([seg[i],-seg[i]])))
        temp_arg=np.argmax(res)
        if res[temp_arg]>scoring:
            scoring=res[temp_arg]
            if temp_arg<seg_len:
                arg+=np.array([seg[temp_arg],0])
            elif temp_arg<2*seg_len:
                arg+=np.array([0,seg[temp_arg-seg_len]])
            elif temp_arg<3*seg_len:
                arg+=np.array([seg[temp_arg-2*seg_len],seg[temp_arg-2*seg_len]])
            else:
                arg+=np.array([seg[temp_arg-3*seg_len],-seg[temp_arg-3*seg_len]])
        else:
            termination=True
    return arg



''' First algorithm for creating a graph : idea is comparing the centroids of the masks of two images '''
#not sure that we need this, naiv graph is better
def graph_centroid(dic,saving=False,savingpath='dict'):
    centroid_parent=[]
    fichier=list(dic.keys())[0]
    dic[list(dic.keys())[-1]]['centroid_child']=[]
    while dic[fichier]['child']!='':
        dic[fichier]['centroid_parent']= centroid_parent 
        child=dic[fichier]['child']
        transfert=dic[child]['main_centroid']-dic[fichier]['main_centroid']
        centro_c=np.copy(dic[child]['centroid'])
        centro_p=np.copy(dic[fichier]['centroid'])
        len_p=len(centro_p)
        for i in range(len_p):
            centro_p[i,:]+=transfert
        centroid_parent ,centroid_child =close_pt_matrix(centro_c,centro_p)
        dic[fichier]['centroid_child']=centroid_child
        fichier=dic[fichier]['child']
    dic[child]['centroid_parent']= centroid_parent 
    if saving:
        np.save(savingpath,dic)

# Computing the closest point between two distributions
@jit 
def close_pt_matrix(points,cloud):
    number_points=len(points)
    number_cloud=len(cloud)
    c_matrix=np.zeros((number_points,number_cloud))
    for i in range(number_points):
        for j in range(number_cloud):
            c_matrix[i,j]=np.linalg.norm(points[i]-cloud[j])
    result_p=np.zeros(number_points)
    result_c=np.zeros(number_cloud)
    for i in range(number_points):
        result_p[i]=(np.argmin(c_matrix[i,:]))
    for i in range(number_cloud):
        result_c[i]=(np.argmin(c_matrix[:,i]))
    return result_p,result_c



''' Computing the parent and child (and grand aprent grand child) relations, we suppose that the first and the last images are both usable with defined masks
We link masks if they can superimpose, depending on the threshold (see function comparision_mask)'''
def relation_mask(dic,threshold,saving=False,savingpath='dict'):
    #computing the parent and child relation
    if len(dic.keys())>=2:
        #initialisation
        fichier=list(dic.keys())[0]
        dic[fichier]['mask_parent']=[]
        dic[list(dic.keys())[-1]]['mask_child']=[]
        
        while dic[fichier]['child']!='':
            child=dic[fichier]['child']
            transfert=dic[fichier]['translation_vector']#dic[child]['main_centroid']-dic[fichier]['main_centroid']
            mask_c=np.copy(dic[child]['masks'])
            area_c=np.copy(dic[child]['area'])
            mask_p=np.copy(dic[fichier]['masks'])
            area_p=np.copy(dic[fichier]['area'])
            mask_c=mask_transfert(mask_c,transfert)
            mask_parent , mask_child =comparision_mask(mask_c,mask_p,area_c,area_p,threshold)
            dic[fichier]['mask_child']=mask_child
            dic[child]['mask_parent']=mask_parent
            fichier=child
            
    #computing the grand parent and grand child relation
    if len(dic.keys())>=3:
        #initialisation
        fichier=list(dic.keys())[0]
        last_fichier=list(dic.keys())[-1]
        dic[fichier]['mask_grand_parent']=[]
        dic[dic[fichier]['child']]['mask_grand_parent']=[]
        dic[last_fichier]['mask_grand_child']=[]
        dic[dic[last_fichier]['parent']]['mask_grand_child']=[]
        
        while dic[dic[fichier]['child']]['child']!='':
            grand_child=dic[dic[fichier]['child']]['child']
            transfert=dic[fichier]['translation_vector']#dic[grand_child]['main_centroid']-dic[fichier]['main_centroid']
            mask_c=np.copy(dic[grand_child]['masks'])
            area_c=np.copy(dic[grand_child]['area'])
            mask_p=np.copy(dic[fichier]['masks'])
            area_p=np.copy(dic[fichier]['area'])
            mask_c=mask_transfert(mask_c,transfert)
            mask_grand_parent , mask_grand_child =comparision_mask(mask_c,mask_p,area_c,area_p,threshold)
            dic[fichier]['mask_grand_child']=mask_grand_child
            dic[grand_child]['mask_grand_parent']=mask_grand_parent
            fichier=dic[fichier]['child']
    if saving:
        np.save(savingpath,dic)

# defining the relation between two mask : we link two masks if there intersection represent a fraction superior to threshold of one of their area
@jit 
def comparision_mask(mask_c,mask_p,area_c,area_p,threshold):
    number_mask_c=len(area_c)
    number_mask_p=len(area_p)
    dim_c=np.shape(mask_c)
    dim_p=np.shape(mask_p)
    result_p=np.zeros(number_mask_p)
    result_c=np.zeros(number_mask_c)
    for i in range(1,number_mask_c+1):
        for j in range(1,number_mask_p+1):
            area=0
            for k in range(min(dim_c[0],dim_p[0])):
                for l in range(min(dim_c[1],dim_p[1])):
                    if mask_c[k,l]==i and mask_p[k,l]==j:
                        area+=1
            if area/area_c[i-1]>=threshold:
                result_c[i-1]=j
            if area/area_p[j-1]>=threshold:
                result_p[j-1]=i
    return(result_c,result_p)
            


''' Creating a graph based on the relations relation_mask between parent and child. Add entry naiv_graph_values and  naiv_masks in the dictionnary'''
def naiv_graph(dic,saving=False,savingpath='dict'):
    #Initialisation
    fichier=list(dic.keys())[0]
    values=np.arange(1,np.max(dic[fichier]['masks'])+1,1)
    dic[fichier]['naiv_graph_values']=values
    dic[fichier]['naiv_masks']=np.copy(dic[fichier]['masks'])
    
    
    while dic[fichier]['child']!='':
        
        child=dic[fichier]['child']
        p_nb=np.max(dic[fichier]['masks']) #number of masks of the parent
        c_nb=np.max(dic[child]['masks']) #number of masks of the child
        dic[fichier]['naiv_graph']=[]
        mat_ij=np.zeros((p_nb,c_nb)) # a matrix representing the links in the graph
        
        #constructing the link between the different states for the graph
        for i in range(p_nb):
            for j in range(c_nb):
                if dic[fichier]['mask_child'][i]==j+1 or dic[child]['mask_parent'][j]==i+1:
                    dic[fichier]['naiv_graph'].append([i+1,j+1])
                    mat_ij[i,j]=1
                    
        #continuity of the values if a cell has one and only one child
        new_values=update_values(values,p_nb,c_nb,mat_ij)
        #constructing new masks for continuity
        new_mask=update_masks(np.copy(dic[child]['masks']),new_values)
        dic[child]['naiv_masks']=new_mask
        
        #updating all values
        dic[child]['naiv_graph_values']=new_values
        values=new_values
        fichier=child
        
    if saving:
        np.save(savingpath,dic)

# returns a vector that represent the link between a mask and a value (integer), concerving a value if there is no division or fusion. The information comes from the matrix mat_ij
@jit
def update_values(values,p_nb,c_nb,mat_ij):
    new_values=np.zeros(c_nb)
    for i in range(p_nb):
        for j in range(c_nb):
            if mat_ij[i,j]==sum(mat_ij[:,j])==sum(mat_ij[i,:])==1:
                new_values[j]=values[i]
    #completing vector with other values
    for i in range(c_nb):
        if new_values[i]==0:
            j=1
            while j in new_values:
                j+=1
            new_values[i]=j
    return new_values

# creating a new mask with changed values
@jit
def update_masks(mask,new_values):
    (l,L)=np.shape(mask)
    for j in range(l):
        for k in range(L):
            if mask[j,k]!=0:
                mask[j,k]=new_values[mask[j,k]-1]
    return mask

       

''' Ploting the images, with the masks overlaid, the label of each mask (integer) and the relation with the following masks'''
def plot_graph_and_masks(dic,graph_name):
    #Initialisation
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        # plot image with masks overlaid
        img = cv2.imread(dic[fichier]['adress'])
        #plt.imshow(img)
        masks=dic[fichier][graph_name+'_masks']
        mask_RGB = plot.mask_overlay(img,masks)
        plt.imshow(mask_RGB)
        # plot the centroids
        centr=dic[fichier]['centroid']
        for i in range(len(centr)):
            plt.plot(centr[i,0], centr[i,1], color='k',marker='o')
            plt.annotate(str(int(dic[fichier][graph_name+'_graph_values'][i])), centr[i,:], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
            main_centroid=dic[fichier]['main_centroid']
            plt.plot(main_centroid[0], main_centroid[1], color='w',marker='o')
        #plot the displacement of the centroid between two images
        next_centr=dic[dic[fichier]['child']]['centroid']
        for link in dic[fichier][graph_name+'_graph']:
            if next_centr[link[1]-1][0]!=centr[link[0]-1][0] or next_centr[link[1]-1][1]!=centr[link[0]-1][1]:
                plt.annotate("", xy=(next_centr[link[1]-1][0], next_centr[link[1]-1][1]), xycoords='data', xytext=(centr[link[0]-1][0], centr[link[0]-1][1]), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='w'))
            else :
                plt.plot(next_centr[link[1]-1][0], next_centr[link[1]-1][1],color='w',marker='o', markersize=0.5)
        plt.show()
        fichier=dic[fichier]['child']
    
    # ploting last image
    img = cv2.imread(dic[fichier]['adress'])
    #plt.imshow(img)
    masks=dic[fichier][graph_name+'_masks']
    mask_RGB = plot.mask_overlay(img,masks)
    plt.imshow(mask_RGB)
    # plot the centroids
    centr=dic[fichier]['centroid']
    for i in range(len(centr)):
        plt.plot(centr[i,0], centr[i,1], color='k',marker='o')
        plt.annotate(str(int(dic[fichier][graph_name+'_graph_values'][i])), centr[i,:], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
        main_centroid=dic[fichier]['main_centroid']
        plt.plot(main_centroid[0], main_centroid[1], color='w',marker='o')








'''Running the different functions'''
data_prep(my_data,crop_up,crop_down,crop_left,crop_right)

run_cel(cropped_data,cel_gpu,cel_model_type,cel_channels,cel_diameter,cel_flow_threshold,cel_cellprob_threshold,segments_path)

main_dict=download_dict(cropped_data,segments_path)

main_parenting(main_dict)

centroid_area(main_dict)

clean_masks(ratio_erasing_masks, main_dict)

mask_displacement(main_dict,cel_diameter)

#convex_hull(main_dict,hull_point)

#graph_centroid(main_dict)

relation_mask(main_dict,surface_thresh)

naiv_graph(main_dict,saving=True,savingpath=saving_path)

plot_graph_and_masks(main_dict,"naiv")


'''
for fichiers in main_dict.keys():
    plot_masks(fichiers,main_dict)
'''

'''
#downloading main dictionnary:

main_dict=np.load(saving_path+'.npy', allow_pickle=True).item()
'''
