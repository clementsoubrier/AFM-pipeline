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
from cellpose import utils, io, models, plot
import re
import copy
from shutil import rmtree #erasing a whole directory





''' Running in a in a reduced dataset? '''
test=False

''' Paths of the data and to save the results'''
#Inputs
#directory of the original dataset composed of a sequence of following pictures of the same bacterias, and with log files with .001 or no extension
dir_name=(1-test)*"dataset/" +test*"datatest/"#name of dir
my_data = "../data/"+dir_name #path of dir
#directory with the usefull information of the logs, None if there is no.
data_log=None #"../data/"+          #path and name

#Temporary directories
#directory of cropped data each image of the dataset is cropped to erase the white frame
cropped_data=(1-test)*"cropped_data/" +test*"cropped_datatest/"
#directory with the usefull info extracted from the logs
cropped_log='log_'+(1-test)*"cropped_data/" +test*"cropped_datatest/"
#directory for output data from cellpose 
segments_path = (1-test)*"cellpose_output/"+test*"cellpose_outputtest/"


#Outputs
#directory of the processed images (every image has the same pixel size and same zoom)
final_data="final_data_"+ dir_name
#Saving path for the dictionnary
saving_path='Main_dictionnary_'+ dir_name[:-1]
#dimension and scale of all the final data
dimension_data='Dimension_'+ dir_name[:-1]




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
cel_diameter_param = 1 # parameter to adjust the expected size (in pixels) of bacteria. Incease if cellpose detects too small masks, decrease if it don't detects small mask/ fusion them. Should be around 1 
cel_flow_threshold = 0.9 
cel_cellprob_threshold=0.0



#erasing small masks that have a smaller relative size :
ratio_erasing_masks=0.1

#number of point of the approximated convex hull (at least 3)
hull_point=4

#fraction of the preserved area to consider child and parent relation for masks
surface_thresh=0.6
#searching distance (pixels) for the optimization algorithm to construct transtion vectors between pictures. 
search_diameter =100 

#colors of the masks
colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]


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




''' Preparation of the data (sorting and cropping). Here data is a file with pictures of the cell (.png), and with the log data. For the cropping, we crop each image with the same number of pixels and dimension'''
def data_prep(data,cropup,cropdown,cropleft,cropright,croppeddata,croppedlog,mydata,log_dir=None): #data : directory of the picture, may contain logs. log_dir : the name of a directory with the logs if it exists
    # Load up a list of input files (name as strings) from our example data. Creating the main dictionnay containing all tha information 
    files = os.listdir(data)
    
    # Sort files by timepoint.
    files.sort(key = natural_keys)    
    
    #clean or create the cropped directory
    if os.path.exists(croppeddata):
        for file in os.listdir(croppeddata):
            os.remove(os.path.join(croppeddata, file))
    else:
        os.makedirs(croppeddata)
    #clean or create the log directory
    if os.path.exists(croppedlog):
        for file in os.listdir(croppedlog):
            os.remove(os.path.join(croppedlog, file))
    else:
        os.makedirs(croppedlog)
    
    # removing the non image or log files and cropping the data to erase the white frame : to do manually depending on the dataset. Saving the cropped images
    for i in range(len(files)):
        
        if (files[i].endswith(".png")):
            img = cv2.imread(mydata+ files[i],0) #pictures in black and white
            if (cropup,cropdown,cropleft,cropright)!=(0,0,0,0):
                cropped = img[cropup:-cropdown,cropleft:-cropright]
            else:
                cropped = img
            cv2.imwrite(croppeddata+files[i], cropped)  
        
            
        #dealing with the log files
        elif (files[i].endswith(".001")) or len(re.split(r'\W+',files[i]))==1:# check files ending with .001 or without any . 
            #open text file in read mode
            text_file = open(data+files[i], "r", errors='ignore' )
            #read whole file to a string
            text = text_file.read()
            
            #close file
            text_file.close()
            
            # get rid of the linebreaks
            text=re.sub(r'\n', ' ', text)

            #selecting the good lines (dimension)
            match0=re.match("^.*Samps/(.*|\n)Scan Line.*$",text).group(1)
            #selecting the good lines (angle)
            match1=re.match("^.*Rotate Ang(.*|\n)Stage X.*$",text).group(1)
            #selection of the numbers
            match0=re.findall(r'\d+\.\d+|\d+',match0)#structure : samps / lines; number of line; aspect ratio 1, aspect ratio 2, scan size 1, scan size 2
            match1=re.findall(r'\d+\.\d+|\d+',match1)
            if len(match0)==6: #dealing with the different format size here the aspect ratio has 2 integer
                for j in range(4):
                    match0[j]=int(match0[j]) 
                match0[4]=float(match0[4])
                match0[5]=float(match0[5])
                res=np.array([match0[5]/match0[2]/match0[1],match0[4]/match0[3]/match0[0],float(match1[0])])#structure vertical len of a pixel, horizontal len of a pixel, r0tation
            else :              #dealing with the different format size here the aspect ratio is a float
                for j in range(2): 
                    match0[j]=int(match0[j])
                for j in range(3): 
                    match0[2+j]=float(match0[2+j])  
                res=np.array([match0[4]/match0[0],match0[3]/match0[0],float(match1[0])])
            name=re.split(r'\W+',files[i])[0] #getting rid of the .001
            #saving the dimension of each picture in the log file
            np.save(croppedlog+name,res)
        else:
            files.remove(files[i])
            
            
            
    if log_dir is not None: #going through the directory with the same algorithm
        files = os.listdir(log_dir)
        files.sort(key = natural_keys) 
        for i in range(len(files)):
            #dealing with the log files
            if (files[i].endswith(".001")) or len(re.split(r'\W+',files[i]))==1:# check files ending with .001 or without any . 
                #open text file in read mode
                text_file = open(data+files[i], "r", errors='ignore' )
                #read whole file to a string
                text = text_file.read()
                
                #close file
                text_file.close()
                
                # get rid of the linebreaks
                text=re.sub(r'\n', ' ', text)

                #selecting the good lines (dimension)
                match0=re.match("^.*Samps/(.*|\n)Scan Line.*$",text).group(1)
                #selecting the good lines (angle)
                match1=re.match("^.*Rotate Ang(.*|\n)Stage X.*$",text).group(1)
                #selection of the numbers
                match0=re.findall(r'\d+\.\d+|\d+',match0)#structure : samps / lines; number of line; aspect ratio 1, aspect ratio 2, scan size 1, scan size 2
                match1=re.findall(r'\d+\.\d+|\d+',match1)
                for j in range(4):
                    match0[j]=int(match0[j]) 
                match0[4]=float(match0[4])
                match0[5]=float(match0[5])
                res=np.array([match0[5]/match0[2]/match0[1],match0[4]/match0[3]/match0[0],float(match1[0])])#structure vertical len of a pixel, horizontal len of a pixel, r0tation
                name=re.split(r'\W+',files[i])[0] #getting rid of the .001
                #saving the dimension of each picture in the log file
                np.save(croppedlog+name,res)
            else:
                files.remove(files[i])
  
#A function which allows input files to be sorted by timepoint.
def natural_keys(text):
    return [ int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text) ]



'''Preparation of the data so that each image has the same dimension '''
def dimension_def(croppeddata,log_dir,finaldata,dimensiondata): #dir of the images, dir of the numpy info of logs
    #determining the main dimension and the best vertical/horizontal precision
    phys_dim=[]
    for file in os.listdir(croppeddata):
        namelog=re.findall(r'\d+',file)[0]+'.npy'
        if namelog in os.listdir(log_dir):
            log_para=np.load(log_dir+namelog)
            dim1,dim2 = np.shape(cv2.imread(croppeddata+file,0))
            phys_dim.append([log_para[0]*dim1,log_para[1]*dim2,log_para[0],log_para[1]])
        
    phys_dim=np.array(phys_dim)
    prec=np.min(phys_dim[:,2:4])
    dim_height=int(np.max(phys_dim[:,0])/prec)
    dim_len=int(np.max(phys_dim[:,1])/prec)
    
    np.save(dimensiondata,np.array([dim_height,dim_len,prec]))#saving dimensions of the images and physical dimension of a pixel for future plots
    
    #clean or create the final directory
    if os.path.exists(finaldata):
        for file in os.listdir(finaldata):
            os.remove(os.path.join(finaldata, file))
    else:
        os.makedirs(finaldata)
    
    for file in os.listdir(croppeddata):
        namelog=re.findall(r'\d+',file)[0]+'.npy'
        if namelog in os.listdir(log_dir):
            log_para=np.load(log_dir+namelog)
            img=cv2.imread(croppeddata+file,0)
            newimage=bili_scale_image(img,log_para[0],log_para[1],dim_height,dim_len,prec,prec)
            cv2.imwrite(finaldata+file, newimage) 
    #erasing the cropped data
    rmtree(croppeddata)
    
# scaling algorithm     
@jit
def bili_scale_image(img,preci_h,preci_l,dim_h,dim_l,precf_h,precf_l):
    newimage=np.zeros((dim_h,dim_l))
    dim_img_h,dim_img_l=np.shape(img)
    ratio_h=preci_h/precf_h
    ratio_l=preci_l/precf_l
    dim_zoom_h=int(ratio_h*dim_img_h)
    dim_zoom_l=int(ratio_l*dim_img_l)
    drift_h=int((dim_h-dim_zoom_h)/2)
    drift_l=int((dim_l-dim_zoom_l)/2)
    for i in range(dim_zoom_h-1):
        for j in range(dim_zoom_l-1):
            
            ih1=int((i/ratio_h))
            ih2=int(((i+1)/ratio_h+precf_h))
            
            if ih1==ih2:
                frac_h=1
            else:
                frac_h=(i/ratio_h-ih1)
            jl1=int((j/ratio_l))
            jl2=int(((j+1)/ratio_l))
            if jl1==jl2:
                frac_l=1
            else:
                frac_h=(j/ratio_l-jl1)
            
            res=frac_h*frac_l*img[ih1,jl1]+(1-frac_h)*frac_l*img[ih2,jl1]+frac_h*(1-frac_l)*img[ih1,jl2]+(1-frac_h)*(1-frac_l)*img[ih2,jl2]
            newimage[i+drift_h,j+drift_l]=res
    return(newimage)
   
    
   
''' Running cellpose and saving the results'''
def run_cel(data_file,gpuval,mod,chan,param_dia,thres,celp,seg,dimensiondata):
    #clean or create the output directory
    if os.path.exists(seg):
        for file in os.listdir(seg):
            os.remove(os.path.join(seg, file))
    else:
        os.makedirs(seg)
        
    # computation of the cell diameter
    scale=np.load(dimensiondata+'.npy')[2]
    dia=2/scale*param_dia
    # Specify that the cytoplasm Cellpose model for the segmentation. 
    model = models.Cellpose(gpu=gpuval, model_type=mod)
    # Loop over all of our image files and run Cellpose on each of them. 
    for fichier in os.listdir(data_file):
        img = io.imread(data_file+fichier)
        masks, flows, st, diams = model.eval(img, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)
        # save results so you can load in gui
        io.masks_flows_to_seg(img, masks, flows, diams, seg+fichier[:-4], chan)



''' Creation of a dictionnary with the entry : adress, time, mask, outlines, mask_error, angle, and erasing the temporary directories'''
def download_dict(finaldata,log_dir,segmentspath):
    files = os.listdir(finaldata)
    dic={}
    # Sort files by timepoint.
    files.sort(key = natural_keys)
    t=0
    for fichier in files:
        fichier=fichier[:-4]
        namelog=re.findall(r'\d+',fichier)[0]+'.npy'
        dic[fichier]={}
        dat = np.load(segmentspath+fichier+'_seg.npy', allow_pickle=True).item()
        dic[fichier]['time']=t
        t+=1
        dic[fichier]['adress']=finaldata+fichier+'.png'
        dic[fichier]['masks']=dat['masks']
        dic[fichier]['outlines']=utils.outlines_list(dat['masks'])
        dic[fichier]['masks_error']=(np.max(dic[fichier]['masks'])==0)
        dic[fichier]['angle']=np.load(log_dir+namelog)[-1]
        
    #deleting temporary dir
    #rmtree(segmentspath)
    #rmtree(log_dir)
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
            (centroid,area)=numba_centroid_area(masks)
            dic[fichier]['centroid']=centroid
            dic[fichier]['area']=area

#to improve computation speed
@jit
def numba_centroid_area(masks):
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
    return(centroid,area)

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
                np.save(savingpath,dic)



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
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        child=dic[fichier]['child']
        shape_1=np.shape(dic[fichier]['masks'])
        shape_2=np.shape(dic[child]['masks'])
        shape_f=(min(shape_1[0],shape_2[0]),min(shape_1[1],shape_2[1]))
        mask_p=main_mask(dic[fichier]['masks'][:shape_f[0],:shape_f[1]])
        mask_c=main_mask(dic[child]['masks'][:shape_f[0],:shape_f[1]])
        vecguess=dic[fichier]['main_centroid']-dic[child]['main_centroid']
        angle=dic[fichier]['angle']-dic[child]['angle']
        if angle==0:
            dic[fichier]['translation_vector']=opt_trans_vec(mask_p,mask_c,rad,vecguess)
        else:
            dim1,dim2=np.shape(mask_p)
            centerpoint=np.array([dim1//2,dim2//2],dtype=int)
            vecguess=rotation_vector(angle,vecguess,centerpoint).astype(int)
            mask_p=rotation_img(angle,mask_p,centerpoint)
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

#rotation of a vector around a point, the angle is in radian
@jit 
def rotation_vector(angle,vec,point):
    mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return point+np.dot(mat,vec-point)

#rotation of an image around a point
@jit 
def rotation_img(angle,img,point):
    dim1,dim2=np.shape(img)
    new_img=np.zeros((dim1,dim2))
    for i in range(dim1):
        for j in range(dim2):
            trans_vec=rotation_vector(-angle,np.array([i,j]),point)#sign of the rotation : definition of the angle in the logs
            i_n,j_n=trans_vec[0],trans_vec[1]
            i_t=int(i_n)
            j_t=int(j_n)
            if 0<=i_t<dim1-1 and 0<=j_t<dim2-1:
                frac_i=i_n-i_t
                frac_j=j_n-j_t
                new_img[i,j]=frac_i*frac_j*img[i_t,j_t]+frac_i*(1-frac_j)*img[i_t,j_t+1]+(1-frac_i)*frac_j*img[i_t+1,j_t]+(1-frac_i)*(1-frac_j)*img[i_t+1,j_t+1]
    return new_img

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



''' Computing the parent and child (and grand parent grand child) relations, we suppose that the first and the last images are both usable with defined masks
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
            
            transfert=dic[fichier]['translation_vector']
            mask_c=np.copy(dic[child]['masks'])
            area_c=np.copy(dic[child]['area'])
            mask_p=np.copy(dic[fichier]['masks'])
            area_p=np.copy(dic[fichier]['area'])
            
            mask_c=mask_transfert(mask_c,transfert)
            
            angle=dic[fichier]['angle']-dic[child]['angle']  
            
            if angle!=0:            #check on real data
                dim1,dim2=np.shape(mask_p)
                centerpoint=np.array([dim1//2,dim2//2],dtype=int)
                mask_p=rotation_img(angle,mask_p,centerpoint)
            
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
            transfert=dic[fichier]['translation_vector']
            mask_c=np.copy(dic[grand_child]['masks'])
            area_c=np.copy(dic[grand_child]['area'])
            mask_p=np.copy(dic[fichier]['masks'])
            area_p=np.copy(dic[fichier]['area'])
            
            mask_c=mask_transfert(mask_c,transfert)
            
            angle=dic[fichier]['angle']-dic[grand_child]['angle']  
            
            if angle!=0:            #check on real data
                dim1,dim2=np.shape(mask_p)
                centerpoint=np.array([dim1//2,dim2//2],dtype=int)
                mask_p=rotation_img(angle,mask_p,centerpoint)
            
                
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

    

    
''' Creating a graph based on the relations relation_mask between parent and child. Add entry basic_graph_values and  basic_masks in the dictionnary. Takes into account only the division and not the fusion of cells. Only the division of one cell into two cells is taken into account.'''
def basic_graph(dic,saving=False,savingpath='dict'):
    #Initialisation
    fichier=list(dic.keys())[0]
    values=np.arange(1,np.max(dic[fichier]['masks'])+1,1)
    dic[fichier]['basic_graph_values']=values
    dic[fichier]['basic_masks']=np.copy(dic[fichier]['masks'])
    
    
    while dic[fichier]['child']!='':
        
        child=dic[fichier]['child']
        p_nb=np.max(dic[fichier]['masks']) #number of masks of the parent
        c_nb=np.max(dic[child]['masks']) #number of masks of the child
        dic[fichier]['basic_graph']=[]
        mat_ij=np.zeros((p_nb,c_nb)) # a matrix representing the links in the graph
        
        #constructing the link between the different states for the graph
        for i in range(p_nb):
            for j in range(c_nb):
                if dic[child]['mask_parent'][j]==i+1:
                    mat_ij[i,j]=1
        for i in range(p_nb):
            if sum(mat_ij[i,:])>2:
                mat_ij[i,:]=np.zeros(c_nb)
        for i in range(p_nb):
            for j in range(c_nb):
                if mat_ij[i,j]==1:
                    dic[fichier]['basic_graph'].append([i+1,j+1])
                    
        #continuity of the values if a cell has one and only one child
        new_values=update_values(values,p_nb,c_nb,mat_ij)
        #constructing new masks for continuity
        new_mask=update_masks(np.copy(dic[child]['masks']),new_values)
        dic[child]['basic_masks']=new_mask
        
        #updating all values
        dic[child]['basic_graph_values']=new_values
        values=new_values
        fichier=child
        
    if saving:
        np.save(savingpath,dic)



''' Ploting the images, with the masks overlaid, the label of each mask (integer) and the relation with the following masks'''
def plot_graph_and_masks(dic,graph_name,maskcol):
    #Initialisation
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        # plot image with masks overlaid
        img = cv2.imread(dic[fichier]['adress'],0)
        #plt.imshow(img)
        masks=dic[fichier][graph_name+'_masks']
        masknumber=np.max(masks)
        colormask=copy.deepcopy(maskcol)
        numbercolor=len(colormask)
        colormask=(masknumber//numbercolor+1)*colormask
        colormask=np.array(colormask[:masknumber])
        mask_RGB = plot.mask_overlay(img,masks,colors=colormask)
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
    img = cv2.imread(dic[fichier]['adress'],0)
    #plt.imshow(img)
    masks=dic[fichier][graph_name+'_masks']
    masknumber=np.max(masks)
    colormask=copy.deepcopy(maskcol)
    numbercolor=len(colormask)
    colormask=(masknumber//numbercolor+1)*colormask
    colormask=np.array(colormask[:masknumber])
    mask_RGB = plot.mask_overlay(img,masks,colors=colormask)
    plt.imshow(mask_RGB)
    # plot the centroids
    centr=dic[fichier]['centroid']
    for i in range(len(centr)):
        plt.plot(centr[i,0], centr[i,1], color='k',marker='o')
        plt.annotate(str(int(dic[fichier][graph_name+'_graph_values'][i])), centr[i,:], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
        main_centroid=dic[fichier]['main_centroid']
        plt.plot(main_centroid[0], main_centroid[1], color='w',marker='o')



''' Representation of the links in the graph '''
def plot_graph(dic,graph_name,maskcol,binary=True):
    #First graph (simple plot)
    links_list=[]
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        child=dic[fichier]['child']
        time=dic[fichier]['time']
        values=dic[fichier][graph_name+'_graph_values']
        next_values=dic[child][graph_name+'_graph_values']
        next_gen=[]
        if not binary:
            for link in dic[fichier][graph_name+'_graph']:
                plt.plot([time,time+1],[values[link[0]-1],next_values[link[1]-1]], color='k')
        else:
            for link in dic[fichier][graph_name+'_graph']:
                next_gen.append([values[link[0]-1],next_values[link[1]-1]])
        
            #more advanced graph : contructing the lines in links_list
            
            if links_list!=[]:
                final_state=links_list[-1][1]
                final_number=links_list[-1][2]
                count=0
                for i in range(len(final_state)):
                    for j in next_gen:
                        if final_state[i]==j[0] and int(final_number[i][0])>count:
                            count=int(final_number[i][0])
            
            else:
                final_state=[]
                final_number=[]
                count=0
                
            #detecting the number of lineage roots fo the next branches
            
            
            
            links=[time,[],[],[]]
            for i in range(len(next_gen)):
                #checking if there is division and differentiating the cases
                next_descendant=0
                for j in range(len(next_gen)):
                    if next_gen[i][0]==next_gen[j][0]:
                        if i<j:
                            next_descendant=1
                        if i>j:
                            next_descendant=-1
                            
                #determining the previous lineage            
                place=-1
                for j in range(len(final_state)):
                    if next_gen[i][0]==final_state[j]:
                        place=j
                if place==-1:
                    count+=1
                    number=str(int(count))
                else :
                    number =final_number[place]
                
                if next_descendant==0:
                    a=''
                if next_descendant==1:
                    a='1'
                if next_descendant==-1:
                    a='0'
                links[1].append(next_gen[i][1])
                links[2].append(number+a)
                links[3].append([number,number+a,next_gen[i][1]])
                
    
            links_list.append(links)
            fichier=child
            
    if not binary :
        plt.show() #ploting the first graph
        
    else : 
        #ploting the second graph
        for element in links_list:
            plot_links=element[-1]
            time= int(element[0])
            len_col=len(maskcol)
            for subelement in plot_links:
                n1=subelement[0]
                n2=subelement[1]
                color=subelement[2]
                number1=0
                for i in range(len(n1)):
                    if i==0:
                        number1+=int(n1[i])
                    else:
                        if int(n1[i])==0:
                            number1+=2**(-i-1)
                        else:
                            number1-=2**(-i-1)
                number2=0
                for i in range(len(n2)):
                    if i==0:
                        number2+=int(n2[i])
                    else:
                        if int(n2[i])==0:
                            number2+=2**(-i-1)
                        else:
                            number2-=2**(-i-1)
                col=maskcol[int(color%len_col-1)]
                plot_col=(col[0]/255,col[1]/255,col[2]/255)
                plt.plot([time,time+1],[number1,number2],color=plot_col)
        plt.show()





'''Running the different functions'''
''''''
data_prep(my_data,crop_up,crop_down,crop_left,crop_right,cropped_data,cropped_log,my_data)

dimension_def(cropped_data,cropped_log,final_data,dimension_data)

run_cel(final_data,cel_gpu,cel_model_type,cel_channels,cel_diameter_param,cel_flow_threshold,cel_cellprob_threshold,segments_path,dimension_data)

main_dict=download_dict(final_data,cropped_log,segments_path)

main_parenting(main_dict)

centroid_area(main_dict)

clean_masks(ratio_erasing_masks, main_dict)

mask_displacement(main_dict,search_diameter)

#convex_hull(main_dict,hull_point)

graph_centroid(main_dict)

relation_mask(main_dict,surface_thresh)

naiv_graph(main_dict,saving=True,savingpath=saving_path)


#downloading main dictionnary:

main_dict=np.load(saving_path+'.npy', allow_pickle=True).item()

basic_graph(main_dict,saving=True,savingpath=saving_path)

plot_graph(main_dict,'basic',colormask)

plot_graph_and_masks(main_dict,'basic',colormask)
''''''

