#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:05:28 2022

@author: c.soubrier
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import transform, io, exposure
import cv2
from pystackreg import StackReg
import pystackreg
from numba import jit
import ot.partial as pt
import scipy as sp
from math import sqrt



def max_contrast(img):
    (l,L)=np.shape(img)
    new_img=np.zeros((l,L))
    for i in range(l):
        for j in range(L):
            if img[i,j]!=0:
                new_img[i,j]=1
    return new_img

    
    
@jit 
def mask_transfert(mask,vector):
    (l1,l2)=np.shape(mask)
    new_mask=np.zeros((l1,l2))
    for i in range(l1):
        for j in range(l2):
            if (0<=i+vector[0]<=l1-1) and (0<=j+vector[1]<=l2-1) and mask[i,j]>0:
                new_mask[int(i+vector[0]),int(j+vector[1])]=mask[i,j]
    return new_mask

@jit 
def score_mask(mask1,mask2):
    (l1,l2)=np.shape(mask1)
    score=0
    for i in range(l1):
        for j in range(l2):
            if mask1[i,j]==1 and mask2[i,j]==1:
                score+=1
    return score
          

#@jit 
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
        
            
@jit
def partial_ot_prep(mask_p,mask_c,area_p,area_c):
    (l,m)=np.shape(mask_p)
    dist_p=np.ones(area_p)
    C_p=np.zeros((area_p,area_p))
    t_1=0
    for i in range(l):
        for j in range(m):
            if mask_p[i,j]>0:
                t_2=0
                for i_prime in range(l):
                    for j_prime in range(m):
                        if mask_p[i_prime,j_prime]>0:
                            C_p[t_1,t_2]=sqrt((i-i_prime)**2+(j-j_prime)**2)
                            t_2+=1
                t_1+=1
            
                
    
    (l,m)=np.shape(mask_c)
    dist_c=np.ones(area_c)
    C_c=np.zeros((area_c,area_c))
    t_1=0
    for i in range(l):
        for j in range(m):
            if mask_c[i,j]>0:
                t_2=0
                for i_prime in range(l):
                    for j_prime in range(m):
                        if mask_c[i_prime,j_prime]>0:
                            C_c[t_1,t_2]=sqrt((i-i_prime)**2+(j-j_prime)**2)
                            t_2+=1
                t_1+=1
    return(C_p,C_c,dist_p,dist_c)

dic=np.load('Main_dictionnary.npy', allow_pickle=True).item()
fichier='03310307_Height' 
child=dic[fichier]['child']
mask_2=np.copy(dic[child]['masks'])#[::10,::10]
area_2=sum(np.copy(dic[child]['area']))
mask_1=np.copy(dic[fichier]['masks'])
area_1=sum(np.copy(dic[fichier]['area']))
(C_p,C_c,dist_p,dist_c)=partial_ot_prep(mask_1,mask_2,int(area_1),int(area_2))
gamma=pt.partial_gromov_wasserstein(C_p[::10,::10],C_c[::10,::10],dist_p[::10],dist_c[::10])
'''          
plt.imshow(max_contrast(mask),'Greys_r')
plt.show()

while dic[fichier]['child']!='':
    child=dic[fichier]['child']
    im1=max_contrast(dic[fichier]['masks'])
    im2=max_contrast(dic[child]['masks'])
    shape_1=np.shape(im1)
    shape_2=np.shape(im2)
    shape_f=(min(shape_1[0],shape_2[0]),min(shape_1[1],shape_2[1]))
    sr = StackReg(StackReg.TRANSLATION)
    im1=im1[:shape_f[0],:shape_f[1]]
    im2=im2[:shape_f[0],:shape_f[1]]
    
    sr.register_transform(im1,im2)
    A=np.int_(sr.get_matrix())
    
    plt.imshow(im1,'Greys_r')
    plt.show()
    plt.imshow(im2,'Greys_r')
    plt.show()
    
    vecguess=dic[child]['main_centroid']-dic[fichier]['main_centroid']
    vec=opt_trans_vec(im1,im2,50,vecguess)
    new_im2=mask_transfert(im2,vec)
    
    for i in range (shape_f[0]):
        for j in range (shape_f[1]):
            vec=np.array([i,j,1])
            vec =np.dot(vec, A.T)
            if 0<=vec[0]<shape_f[0] and 0<=vec[1]<shape_f[1]:
                a0=vec[0]
                a1=vec[1]
                new_im2[a0,a1]=im2[i,j]
          
    superp=np.zeros(shape_f)
    for i in range (shape_f[0]):
        for j in range (shape_f[1]):
            if im1[i,j]==1 and new_im2[i,j]==0:
                superp[i,j]=3
            elif im1[i,j]==1 and new_im2[i,j]==1:
                superp[i,j]=2
            elif im1[i,j]==0 and new_im2[i,j]==1:
                superp[i,j]=1
    
    plt.imshow(superp,'Greys_r')
    plt.show()
    fichier=child
'''      