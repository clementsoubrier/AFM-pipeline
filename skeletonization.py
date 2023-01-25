#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:18:34 2022

@author: c.soubrier
"""

#Skeletonization
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from fil_finder import FilFinder2D
from cellpose import utils
from numba import jit
import cv2

@jit
def find_extremal_point(skel):
    (l,L)= np.shape(skel)
    res=np.zeros((2,2))
    pos=0
    for i in range(l):
        for j in range(L):
            count=0
            for k in range(3):
                for p in range(3):
                    if 0<=i-1+k<l and 0<=j-1+p<L:
                        count+=skel[i-1+k,j-1+p]
            if count==2 and skel[i,j]==1:
                res[pos,:]=np.array([i,j])
                pos+=1
    return res


@jit       
def skel_outlines(skel):
    new_point=find_extremal_point(skel)[0,:]
    out_len=np.count_nonzero(skel== 1)
    res=np.zeros((out_len,2))
    res[0,:]=new_point
    for k in range(3):
        for p in range(3):
            np1,np2=int(new_point[0]),int(new_point[1])
            if 0<=np1-1+k<l and 0<=np2-1+p<L:
                if skel[np1-1+k,np2-1+p]==1 and (k,p)!=(1,1):
                    old_point,new_point=new_point, np.array([np1-1+k,np2-1+p])
    res[1,:]=new_point
    for i in range(2,out_len):
        for k in range(3):
            for p in range(3):
                np1,np2=int(new_point[0]),int(new_point[1])
                op1,op2=int(old_point[0]),int(old_point[1])
                if 0<=np1-1+k<l and 0<=np2-1+p<L:
                    if skel[np1-1+k,np2-1+p]==1 and (k,p)!=(1,1) and (np1-1+k,np2-1+p)!=(op1,op2):
                        old_point,new_point=new_point, np.array([new_point[0]-1+k,new_point[1]-1+p])
        res[i,:]=new_point
    return res
    

dic=np.load('Main_dictionnary.npy', allow_pickle=True).item()
'''
fichier=list(dic.keys())[36]
masks=dic[fichier]['masks']
(l,L)= np.shape(masks)
newmask=np.zeros((l,L))
for i in range(l):
    for j in range(L):
        if masks[i,j]==1:
            newmask[i,j]=1
fil=FilFinder2D(newmask,mask=newmask)
fil.preprocess_image(skip_flatten=True)     
fil.create_mask(use_existing_mask=True)
fil.medskel(verbose=False)  
fil.analyze_skeletons(skel_thresh=60*u.pix,branch_thresh=40*u.pix,prune_criteria='length')
newmask=0.5*newmask
skelet = np.int_(fil.skeleton)
plt.imshow(newmask+skelet,'Greys_r')
A=find_extremal_point(skelet)
for i in range(len(A)):
    plt.plot(A[i][1],A[i][0],color='b',marker='o')


out=skel_outlines(skelet)
plt.plot(out[:,1],out[:,0], color='b')
plt.show()
'''
fichier=list(dic.keys())[60]
while dic[fichier]['child']!='':
    child=dic[fichier]['child']
    masks=dic[fichier]['naiv_masks']
    (l,L)= np.shape(masks)
    newmask=np.zeros((l,L))
    for i in range(l):
        for j in range(L):
            if masks[i,j]==1:
                newmask[i,j]=1
    fil=FilFinder2D(newmask,mask=newmask)
    fil.preprocess_image(skip_flatten=True)     
    fil.create_mask(use_existing_mask=True)
    fil.medskel(verbose=False)  
    fil.analyze_skeletons(skel_thresh=25*u.pix,branch_thresh=20*u.pix,prune_criteria='length')
    newmask=0.5*newmask
    skelet = np.int_(fil.skeleton)
    out=skel_outlines(skelet)
    img = cv2.imread(dic[fichier]['adress'],0)
    plt.imshow(img,'Greys_r')
    plt.plot(out[:,1],out[:,0], color='b')
    plt.show()
    plt.plot([img[int(out[i,0]),int(out[i,1])] for i in range(len(out))])
    plt.show()
    fichier=child
