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
from skimage.morphology import skeletonize #,medial_axis
from PIL import Image #dealing with .tif images
import scipy





''' Paths of the data and to save the results'''


# =============================================================================
# #Inputs
# #directory of the original dataset composed of a sequence of following pictures of the same bacterias, and with log files with .001 or no extension
# Main_directory=   #the directory you chos to work on
# dir_name=Main_directory+"Height/"             #name of dir
# my_data = "../data/"+dir_name       #path of dir
# #directory with the usefull information of the logs, None if there is no.
# data_log="../data/"+Main_directory+"log/"             #       #path and name
# 
# #Temporary directories
# #directory of cropped data each image of the dataset is cropped to erase the white frame
# cropped_data=dir_name+"cropped_data/" 
# #directory with the usefull info extracted from the logs
# cropped_log=dir_name+"log_cropped_data/" 
# #directory for output data from cellpose 
# segments_path = dir_name+"cellpose_output/"
# 
# 
# #Outputs
# #directory of the processed images (every image has the same pixel size and same zoom)
# final_data=dir_name+"final_data/" 
# #dictionnary and dimension directory
# Dic_dir=dir_name+"Dic_dir/" 
# #Saving path for the dictionnary
# saving_path=Dic_dir+'Main_dictionnary'
# #dimension and scale of all the final data
# dimension_data=Dic_dir+'Dimension'
# =============================================================================


Directory=  "dataset/"#"WT_mc2_55/30-03-2015/" #the directory you chose to work on
# different type of datassets with their cropping parameter
data_set=[["dataset/",True],["delta_3187/21-02-2019/",True],["delta_3187/19-02-2019/",True],["delta_parB/03-02-2015/",False],["delta_parB/15-11-2014/",False],["delta_parB/18-01-2015/",False],["delta_parB/18-11-2014/",False],["delta_lamA_03-08-2018/1/",True],["delta_lamA_03-08-2018/2/",True],["WT_mc2_55/06-10-2015/",False],["WT_mc2_55/05-10-2015/",False],["WT_mc2_55/30-03-2015/",True],["WT_mc2_55/05-02-2014/",False],["WT_11-02-15/",False,False],["delta_ripA/14-10-2016/",False],["delta_ripA/160330_rip_A_no_inducer/",True],["delta_ripA/160407_ripA_stiffness_septum/",True],["delta_LTD6_04-06-2017/",False]  ]   #Anti"WT_mc2_55/30-03-2015/","WT_mc2_55/05-02-2014/","WT_11-02-15/",      #Yes "delta_3187/21-02-2019/","delta_3187/19-02-2019/","delta parB/03-02-2015/","delta parB/15-11-2014/","delta parB/18-11-2014/","delta_lamA_03-08-2018/1/","delta_lamA_03-08-2018/2/"          #Maybe "delta parB/18-01-2015/","WT_mc2_55/04-06-2016/","WT_mc2_55/06-10-2015/","WT_mc2_55/05-10-2015/","delta_LTD6_04-06-2017/"                #No "delta ripA/14-10-2016/","delta ripA/160330_rip_A_no_inducer/","delta ripA/160407_ripA_stiffness_septum/","Strep_pneumo_WT_29-06-2017/","Strep_pneumo_WT_07-07-2017/"

Bad_quality_data=[["WT_mc2_55/04-06-2016/1/",False],["WT_mc2_55/04-06-2016/2/",False],["WT_mc2_55/04-06-2016/3/",False],["Strep_pneumo_WT_29-06-2017/",True,False],["Strep_pneumo_WT_07-07-2017/",True]]

Non_usable_data=["delta_murT","WT_mc2_55/03-09-2014"]

result_path='Height/Dic_dir/'

dic_name='Main_dictionnary.npy'

dim_name='Dimension.npy'

color_mask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]







''' main dictionnary main_dict: each image contain a dictionnary with
time : the time of the image beginning at 0
adress : location of the cropped file
masks : the masks as numpy array
outlines : the outlines as a list of points
angle: rotation angle of the image compared to a fixed reference
centroid : an array containing the position of the centroid of each mask the centroid[i] is the centroid of the mask caracterized by i+1
area : an array containing the position of the area (number of pixels) of each mask
mask_error : True if cellpose computes no mask, else False
convex_hull : convex hull of each mask
main_centroid : the centroid of all masks
parent / child : previous / next acceptable  file
mask_parent/child/grand_parent/grand_child : tracking of the mask (area changes) over 1 and 2 generations
translation_vector : vector to superimpose parent and child picture
graph_name+'_graph' : graph representing the relation between masks
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
        
        elif (files[i].endswith(".tif")):
            img = np.array(Image.open(mydata+files[i]))
            cv2.imwrite(croppeddata+files[i][:-3]+'png', img)
            
   
            
    if log_dir is not None: #going through the directory with the same algorithm
        files = os.listdir(log_dir)
        files.sort(key = natural_keys) 
        for i in range(len(files)):
            #dealing with the log files
            if (files[i].endswith(".001")) or len(re.split(r'\W+',files[i]))==1:# check files ending with .001 or without any . 
                #open text file in read mode
                text_file = open(log_dir+files[i], "r", errors='ignore' )
                #read whole file to a string
                text = text_file.read()
                
                #close file
                text_file.close()
                
                # get rid of the linebreaks
                text=re.sub(r'\n', ' ', text)

                #selecting the good lines (dimension)
                first_mat=re.match("^.*Samps/(.*|\n)Scan Line.*$",text)
                first_mat2=re.match("^.*Scan size(.*?|\n)~m.*$",text) 
                first_mat3=re.match("^.*Scan size(.*?|\n)nm.*$",text) 
                if first_mat is not None:
                    match0=first_mat.group(1)
                    #selecting the good lines (angle)
                    match1=re.match(r"^.*Rotate Ang(.*?|\n)\\.*$",text).group(1)
                    #selection of the numbers
                    match0=re.findall(r'\d+\.\d+|\d+',match0)#structure : samps / lines; number of line; aspect ratio 1, aspect ratio 2, scan size 1, scan size 2
                    match1=re.findall(r'\d+\.\d+|\d+',match1)
                    if len(match0)>=6:
                        for j in range(4):
                            match0[j]=int(match0[j]) 
                        match0[4]=float(match0[4])
                        match0[5]=float(match0[5])
                        res=np.array([match0[5]/match0[2],match0[4]/match0[3],float(match1[0])])#structure vertical len of the image, horizontal len the image, r0tation
                        name=re.split(r'\W+',files[i])[0] #getting rid of the .001
                        #saving the dimension of each picture in the log file
                        np.save(croppedlog+name,res)
                elif first_mat2 is not None:
                    first_mat=re.findall(r'\d+\.\d+|\d+',first_mat2.group(1))
                    second_mat=re.match(r"^.*Aspect ratio(.*?|\n)\\.*$",text).group(1)
                    second_mat=re.findall(r'\d+\.\d+|\d+',second_mat)
                    if len(second_mat)==2:
                        match1=re.match(r"^.*Rotate Ang(.*?|\n)\\.*$",text).group(1)
                        match1=re.findall(r'\d+\.\d+|\d+',match1)
                        if len(first_mat)==2:
                            match0=[0,0,int(second_mat[0]),int(second_mat[1]),float(first_mat[0]),float(first_mat[1])]
                        else:
                            match0=[0,0,int(second_mat[0]),int(second_mat[1]),float(first_mat[0]),float(first_mat[0])]
                        res=np.array([match0[5]/match0[2],match0[4]/match0[3],float(match1[0])])#structure vertical len of the image, horizontal len the image, r0tation
                        name=re.split(r'\W+',files[i])[0] #getting rid of the .001
                        #saving the dimension of each picture in the log file
                        np.save(croppedlog+name,res)
                elif first_mat3 is not None:
                    first_mat=re.findall(r'\d+\.\d+|\d+',first_mat3.group(1))
                    second_mat=re.match(r"^.*Aspect ratio(.*?|\n)\\.*$",text).group(1)
                    second_mat=re.findall(r'\d+\.\d+|\d+',second_mat)
                    if len(second_mat)==2:
                        match1=re.match(r"^.*Rotate Ang(.*?|\n)\\.*$",text).group(1)
                        match1=re.findall(r'\d+\.\d+|\d+',match1)
                        if len(first_mat)==2:
                            match0=[0,0,int(second_mat[0]),int(second_mat[1]),float(first_mat[0])/1000,float(first_mat[1])/1000]
                        else:
                            match0=[0,0,int(second_mat[0]),int(second_mat[1]),float(first_mat[0])/1000,float(first_mat[0])/1000]
                        res=np.array([match0[5]/match0[2],match0[4]/match0[3],float(match1[0])])#structure vertical len of the image, horizontal len the image, r0tation
                        name=re.split(r'\W+',files[i])[0] #getting rid of the .001
                        #saving the dimension of each picture in the log file
                        np.save(croppedlog+name,res)

  
#A function which allows input files to be sorted by timepoint.
def natural_keys(text):
    return [ int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text) ]



'''Preparation of the data so that each image has the same dimension '''
def dimension_def(croppeddata,log_dir,finaldata,dimensiondata,dimension_directory): #dir of the images, dir of the numpy info of logs

    #clean or create the final directory
    if os.path.exists(finaldata):
        for file in os.listdir(finaldata):
            os.remove(os.path.join(finaldata, file))
    else:
        os.makedirs(finaldata)
        
    if os.path.exists(dimension_directory):
        for file in os.listdir(dimension_directory):
            os.remove(os.path.join(dimension_directory, file))
    else:
        os.makedirs(dimension_directory)
    
        
    #determining the main dimension and the best vertical/horizontal precision
    phys_dim=[]
    for file in os.listdir(croppeddata):
        namelog=re.findall(r'\d+',file)[0]+'.npy'
        if namelog in os.listdir(log_dir):
            log_para=np.load(log_dir+namelog)
            dim1,dim2 = np.shape(cv2.imread(croppeddata+file,0))
            phys_dim.append([log_para[0],log_para[1],log_para[0]/dim1,log_para[1]/dim2])
    phys_dim=np.array(phys_dim)
    prec=np.min(phys_dim[:,2:4])
    dim_height=int(np.max(phys_dim[:,0])/prec)
    dim_len=int(np.max(phys_dim[:,1])/prec)
    print('dimensions :'+str(dim_height)+', '+str(dim_len))
    np.save(dimensiondata,np.array([dim_height,dim_len,prec]))#saving dimensions of the images and physical dimension of a pixel for future plots
    
    
    for file in os.listdir(croppeddata):
        namelog=re.findall(r'\d+',file)[0]+'.npy'
        if namelog in os.listdir(log_dir):
            log_para=np.load(log_dir+namelog)
            img=cv2.imread(croppeddata+file,0)
            dim1,dim2 = np.shape(img)
            newimage=bili_scale_image(img,log_para[0]/dim1,log_para[1]/dim2,dim_height,dim_len,prec,prec)
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
def run_cel(data_file,gpuval,mod,chan,param_dia,thres,celp,seg,dimensiondata,batch_size=8):
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
        masks, flows, st, diams = model.eval(img,batch_size, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)
        # save results so you can load in gui
        io.masks_flows_to_seg(img, masks, flows, diams, seg+fichier[:-4], chan)



''' Creation of a dictionnary with the entry : adress, time, mask, mask_error, angle, and erasing the temporary directories'''
def download_dict(finaldata,log_dir,segmentspath,dimensiondata):
    files = os.listdir(finaldata)
    dic={}
    dimen=np.load(dimensiondata+'.npy')
    
    
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
        dic[fichier]['masks']=dat['masks'].astype(np.int32)
        dic[fichier]['outlines']=utils.outlines_list(dat['masks'])
        dic[fichier]['masks_error']=(np.max(dic[fichier]['masks'])==0)
        dic[fichier]['angle']=np.load(log_dir+namelog)[-1]
        dic[fichier]['resolution']=dimen[2]
        
    #deleting temporary dir
    rmtree(segmentspath)
    rmtree(log_dir)
    os.remove(dimensiondata+'.npy')
    return dic



''' Construction the sequence of usable pictures : linking images with previous (parent) and following (child) image'''
def main_parenting(dic):
    parent=''
    list_key=list(dic.keys())
    for fichier in list_key:
        if not dic[fichier]['masks_error']:
            dic[fichier]['parent']=parent
            parent=fichier
        else :
            del dic[fichier]
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
    centroid=np.zeros((mask_number,2),dtype=np.int32)
    area=np.zeros(mask_number,dtype=np.int32)
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
        centroid[i,:]=np.array([vec2//count,vec1//count],dtype=np.int32)
    return(centroid,area)

''' Erasing too small masks (less than the fraction frac_mask of the largest mask), and mask with a ratio of saturated area superior to frac_satur . Creating the centroid of the union of acceptable  mask and saving as main_centroid'''
def clean_masks(frac_mask,frac_satur,dic,saving=False,savingpath='dict'): 
    #Erase the masks that are too small (and the centroids too)
    for fichier in dic.keys():
        if not dic[fichier]['masks_error']:
            masks=dic[fichier]['masks']
            img=cv2.imread(dic[fichier]['adress'],0)
            area=dic[fichier]['area']
            centroid=dic[fichier]['centroid']
            outlines=dic[fichier]['outlines']
            
            
            max_area=np.max(area)
            L=len(area)
            non_saturated=np.zeros(L) #detection of the non saturated masks
            for i in range(L):
                if satur_area(img,masks,i+1)<frac_satur*area[i]:
                    non_saturated[i]=1
            
            max_area=max([area[i]*non_saturated[i] for i in range(L)])
            non_defect=np.zeros(L) #classification of the allowed masks
            non_defect_count=0
            newoutlines=[]
            
            
            for i in range(L):
                if area[i]>=frac_mask*max_area and non_saturated[i]==1:
                    non_defect_count+=1
                    non_defect[i]=non_defect_count
                    newoutlines.append(outlines[i][:,::-1].astype(np.int32))
                    
            #update the outlines
            dic[fichier]['outlines']=newoutlines
            #new value of the area and the centroid
            area2=np.zeros(non_defect_count).astype(np.int32)
            centroid2=np.zeros((non_defect_count,2),dtype=np.int32)
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
            if sum(area2)>0:
                main_centroid0=0
                main_centroid1=0
                for i in range (non_defect_count):
                    main_centroid0+=area2[i]*centroid2[i,0]
                    main_centroid1+=area2[i]*centroid2[i,1]
                
                dic[fichier]['main_centroid']=np.array([main_centroid0//sum(area2),main_centroid1//sum(area2)],dtype=np.int32)
            else :
                dic[fichier]['masks_error']=True
    if saving:
        np.save(savingpath,dic)


@jit
def satur_area(img,masks,number):
    (m,n)=np.shape(masks)
    area=0
    for i in range(m):
        for j in range(n):
            if masks[i,j]==number and img[i,j]==255:
                area+=1
    return area
 


'''Contructing a list containing all the masks of the dataset with this structure : list_index,dataset, frame,mask_index'''

def construction_mask_list(dic,dataset,listsavingpath):
    index=0
    final_list=[]
    for fichier in dic.keys():
        if not dic[fichier]['masks_error']:
            masks=dic[fichier]['masks']
            mask_number=np.max(masks)
            list_index=[]
            for i in range(mask_number):
                final_list.append([index,dataset,fichier,i+1])
                list_index.append(index)
                index+=1
            dic[fichier]["mask_list"]=list_index
    np.save(listsavingpath,np.array(final_list,dtype=object))




''' Computing the centerline of each mask and saving them in the dictionnary'''
#main function to save the centerlines in the dictionnary
def centerline_mask(dic,alpha_erasing_branch,saving=False,savingpath='dict'):

    for fichier in dic.keys():
        if not dic[fichier]['masks_error']:
            masks=dic[fichier]['masks']
            mask_number=np.max(masks)
            centerlines=[]
            for i in range(mask_number):
                mask_i=transfo_bin(masks,i+1) #isolating the i+1-th mask
                centerlines.append(construc_center(mask_i,alpha_erasing_branch))
            dic[fichier]['centerlines']=centerlines
    if saving:
        np.save(savingpath,dic)

#isolating the mask with number num
@jit
def transfo_bin(mask,num):
    (dim1,dim2)=np.shape(mask)
    newmask=np.zeros((dim1,dim2),dtype=np.int32)
    for i in range(dim1):
        for j in range(dim2):
            if mask[i,j]==num:
                newmask[i,j]=1
    return newmask

#Constructing the centerline given a unique mask
def construc_center(mask,alpha):
    skel=padskel(mask)              #runing the skeleton algorithm and obtaining a boolean picture
    ex,div=find_extremal_division_point(skel)               #extracting the extremal and division points of the skel
    extr_seg,int_seg=division_path_detek(skel)          #EXTRACTING THE DIFFERENT SEGMENTS
    extr_seg=fusion_erasing(extr_seg,ex,div,alpha)          #fusioning/erasing the external segments if needed
    len_int=len(int_seg)
    len_extr=len(extr_seg)
    
    if len_int==0:
        if len_extr==1:
            return np.array(extr_seg[0]).astype(np.int32)
        elif len_extr==2:
            compo1=extr_seg[0]
            compo2=extr_seg[1][-1::-1]
            if compo1[-1]==compo2[0]:
                del compo1[-1]
            sol=compo1+list(linear_interpolation(compo1[-1],compo2[0]))+compo2
            return np.array(sol).astype(np.int32)
        else:
            print("Error skel format,len_int==0,len_extr="+str(len_extr))
            return np.array([[]],dtype=np.int32)
        
        
    elif len_int==1 and len_extr==2:
        compo1=extr_seg[0]
        compo3=extr_seg[0][::-1]
        if max(abs(compo1[-1][0]-int_seg[0][0][0]),abs(compo1[-1][1]-int_seg[0][0][1]))<=3:
            compo2=int_seg[0]
        else:
            compo2=int_seg[0][::-1]
        if compo1[-1]==compo2[0]:
            del compo1[-1]
        if compo2[-1]==compo3[0]:
            del compo2[-1]
        sol=compo1+list(linear_interpolation(compo1[-1],compo2[0]))+compo2+list(linear_interpolation(compo2[-1],compo3[0]))+compo3
        return np.array(sol).astype(np.int32)
    
        
    #we can do the case with 2 internal segments
    elif len_int==2 and len_extr==3:
        int0=int_seg[0]
        int1=int_seg[1]
        
        #selecting the intersection point between the two internal branches
        if max(abs(int0[0][0]-int1[0][0]),abs(int0[0][1]-int1[0][1]))<=3:
            point=int0[0]
            new_int_seg1=int0[-2::-1]
            new_int_seg2=int1
        elif max(abs(int0[0][0]-int1[-1][0]),abs(int0[0][1]-int1[-1][1]))<=3:
            point=int0[0]
            new_int_seg1=int1[:-1]
            new_int_seg2=int0
        elif max(abs(int0[-1][0]-int1[-1][0]),abs(int0[-1][1]-int1[-1][1]))<=3:
            point=int0[-1]
            new_int_seg1=int0
            new_int_seg2=int1[-2::-1]
        else :
            point=int0[-1]
            new_int_seg1=int0
            new_int_seg2=int1[1:]
        #simplifying by erasing the external branch linked to both internal branch
        for elem in extr_seg:
            if max(abs(elem[-1][0]-point[0]),abs(elem[-1][1]-point[1]))<=3:
                extr_seg.remove(point)
        #previous case len_int==1 and len_extr==2
        if max(abs(extr_seg[1][-1][0]-new_int_seg1[0][0]),abs(extr_seg[1][-1][1]-new_int_seg1[0][1]))<=3:
            new_int_seg1,new_int_seg2=copy.deepcopy(new_int_seg2[::-1]),copy.deepcopy(new_int_seg1[::-1])
        if extr_seg[0][-1]==new_int_seg1[0]:
            del new_int_seg1[0]
        if new_int_seg1[-1]==new_int_seg2[0]:
            del new_int_seg2[0]
        if new_int_seg2[-1]==extr_seg[1][-1]:
            del extr_seg[1][-1]
        sol=extr_seg[0]+list(linear_interpolation(extr_seg[0][-1],new_int_seg1[0]))+new_int_seg1+list(linear_interpolation(new_int_seg1[-1],new_int_seg2[0]))+new_int_seg2+list(linear_interpolation(new_int_seg2[-1],extr_seg[1][-1]))+extr_seg[1][::-1]
        
        return np.array(sol)
    else:
        print("Error skel format, len_int="+str(len_int)+"len_extr="+str(len_extr))
        return np.array([[]],dtype=np.int32)
    
              #construct an integer line between 2 points in 2 D
def linear_interpolation(point1,point2): 
    len_list= max(np.abs(point1[0]-point2[0]),np.abs(point1[1]-point2[1]))-1
    if len_list<=0:
        return np.zeros(0)
    else :
        if np.abs(point1[0]-point2[0])+1==len_list:
            val1=np.linspace(point1[0]+1,point2[0]-1,len_list,dtype=np.int32)
            val2=np.linspace(point1[1],point2[1],len_list,dtype=np.int32)
        else:
            val1=np.linspace(point1[0],point2[0],len_list,dtype=np.int32)
            val2=np.linspace(point1[1]+1,point2[1]-1,len_list,dtype=np.int32)
        
        final=np.zeros((len_list,2),dtype=np.int32)
        final[:,0]=val1
        final[:,1]=val2
        return final

# skeletonization algorithm
def padskel(mask):
    ''' 
    Runs skimage.morphology.skeletonize on a padded version of the mask, to avoid errors.
    
    Parameters
    ----------
    mask = an opencv image
    
    Returns
    -------
    skel = a skeleton (boolean array)
    '''
    mask = cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,None,value=0)#add a border to the skel
    skel = skeletonize(mask,method='lee')              #best skeletonization algorithms
    #skel = skeletonize(mask)           #second skeletonization algorithms
    #skel= medial_axis(mask)            #third skeletonization algorithms
    skel = skel[20:np.shape(skel)[0]-20,20:np.shape(skel)[1]-20]
    return skel

# Because the skeleton is 1D, the extremal points have only one neighbour, the division points have 3, we don't take into account the cases with 4 or more neighbour (bad quality skeleton)
def find_extremal_division_point(skel):
    ex=[]
    div=[]
    dim1,dim2=np.shape(skel)
    for i in range(dim1):
        for j in range(dim2):
            if skel[i,j]:
                count=neighbourcount(i,j,dim1,dim2,skel)
                if count==1:
                    ex.append([i,j])
                elif count==3:
                    div.append([i,j])
    return ex,div


#counting the neighbour of a point that are also in the skeleton
@jit
def neighbourcount(i,j,dim1,dim2,skel):
    count=0
    for k in [-1,0,1]:
        for l in [-1,0,1]:
            if (k,l)!=(0,0) and 0<=i+k<dim1 and 0<=j+l<dim2:
                if skel[i+k,j+l]:
                    count+=1
    return count                  

#computing the different segment composing the skeleton : int_seg the list of the internal segment (between two points of div) the list of the external segment (between a points of ext and another point). Each segment is a list of points.
def division_path_detek(skel):#computing the different segment composing the skeleton : int_seg the list of the internal segment (between two points of div) the list of the external segment (between a points of ext and another point)
    ex,div=find_extremal_division_point(skel)
    dim1,dim2=np.shape(skel)
    if len(ex)<=1 :
        return [],[]
    elif len(ex)==2 : #simple case without branching
        old_point=ex[0]
        tot_seg=[old_point]
        for k in [-1,0,1]:
            for l in [-1,0,1]:
                if (k,l)!=(0,0) and 0<=old_point[0]+k<dim1 and 0<=old_point[1]+l<dim2:
                    if skel[old_point[0]+k,old_point[1]+l]:
                        new_point=[old_point[0]+k,old_point[1]+l]
                        tot_seg.append(new_point)
        while new_point not in ex :
            pot_new_point=[]
            for k in [-1,0,1]:
                for l in [-1,0,1]:
                    if (k,l)!=(0,0) and 0<=new_point[0]+k<dim1 and 0<=new_point[1]+l<dim2 and (new_point[0]+k,new_point[1]+l)!=(old_point[0],old_point[1]) and skel[new_point[0]+k,new_point[1]+l]:
                        pot_new_point=[new_point[0]+k,new_point[1]+l]
            if pot_new_point==[]:
                break
            old_point,new_point=new_point,pot_new_point
            tot_seg.append(new_point)         
        return [tot_seg],[]
    
    elif len(ex)==3 :#simple case three connected points
        extr_seg=[] #computation of the external case
        for i in range(3):
            old_point=ex[i]
            skel[old_point[0],old_point[1]]=False
            new_seg=[old_point]
            for k in [-1,0,1]:
                for l in [-1,0,1]:
                    if  0<=old_point[0]+k<dim1 and 0<=old_point[1]+l<dim2:
                        if skel[old_point[0]+k,old_point[1]+l]:
                            new_point=[old_point[0]+k,old_point[1]+l]
                            new_seg.append(new_point)
                            skel[new_point[0],new_point[1]]=False  
            while (new_point not in div) :
                pot_new_point=[]
                for k in [-1,0,1]:
                    for l in [-1,0,1]:
                        if  0<=new_point[0]+k<dim1 and 0<=new_point[1]+l<dim2 and skel[new_point[0]+k,new_point[1]+l]:
                                pot_new_point=[new_point[0]+k,new_point[1]+l]
                if pot_new_point==[]:
                    break
                old_point,new_point=new_point,pot_new_point
                if new_point not in div:
                    new_seg.append(new_point)
                    skel[new_point[0],new_point[1]]=False 
                else:
                    close_neighbour=False
                    for elem in div:
                        if max(abs(new_point[0]-elem[0]),abs(new_point[1]-elem[1]))==1:
                            close_neighbour=True
                    if close_neighbour:
                        new_seg.append(new_point)
                        skel[new_point[0],new_point[1]]=False 
                    else:
                        new_seg.append(new_point)
            extr_seg.append(new_seg)
        return extr_seg,[]
    
    
     
    else: #case with branching
        extr_seg=[] #computation of the external case
        for i in range(len(ex)):
            old_point=ex[i]
            skel[old_point[0],old_point[1]]=False
            new_seg=[old_point]
            for k in [-1,0,1]:
                for l in [-1,0,1]:
                    if  0<=old_point[0]+k<dim1 and 0<=old_point[1]+l<dim2:
                        if skel[old_point[0]+k,old_point[1]+l]:
                            new_point=[old_point[0]+k,old_point[1]+l]
                            new_seg.append(new_point)
                            skel[new_point[0],new_point[1]]=False  
            while (new_point not in div) :
                pot_new_point=[]
                for k in [-1,0,1]:
                    for l in [-1,0,1]:
                        if  0<=new_point[0]+k<dim1 and 0<=new_point[1]+l<dim2 and skel[new_point[0]+k,new_point[1]+l]:
                                pot_new_point=[new_point[0]+k,new_point[1]+l]
                if pot_new_point==[]:
                    break
                old_point,new_point=new_point,pot_new_point
                if new_point not in div:
                    new_seg.append(new_point)
                    skel[new_point[0],new_point[1]]=False 
                else:
                    close_neighbour=False
                    for elem in div:
                        if max(abs(new_point[0]-elem[0]),abs(new_point[1]-elem[1]))<=2:
                            close_neighbour=True
                    if close_neighbour:
                        new_seg.append(new_point)
                        skel[new_point[0],new_point[1]]=False 
                    else:
                        new_seg.append(new_point)
            extr_seg.append(new_seg)
            
        newextr_seg,newint_seg=division_path_detek(skel)
        return extr_seg, newextr_seg+newint_seg



#simplify a segment into a 1D line (so that each pixel has at most 2 neighbour), we suppose that the segment goes in one direction ( it can't go back to point where it was, but can stay one the same pixel). This function is only used in simple_fusion
def one_D_line(seg) : 
    #selecting the two first element
    newseg=[seg[0],seg[1]]
    if 2<len(seg):
        for j in range(2,len(seg)):
            parent=newseg[-1]
            grandparent=newseg[-2]
            child=seg[j]
            if max(abs(child[0]-parent[0]),abs(child[1]-parent[1]),abs(child[0]-grandparent[0]),abs(child[1]-grandparent[1]))<=1:
                del newseg[-1]
                newseg.append(child)
            else :
                newseg.append(child)
    if newseg[-1]==newseg[-2]:
        del newseg[-1]
    return newseg
 
# Erasing the shortest branch if it is alpha>1 shorter than the longuest. Else fusioning two external segments linked to the same division point to get just a 1D line, works on simple cases only
def fusion_erasing(extr_seg,ex,div,alpha):
    if len(extr_seg)==1: #case with just one segment
        return extr_seg
    else: #each external segment has a connection to a division point
        new_seg=[]
        newdiv=[] #selection only one of the three division point if three are touching
        
        for point in div:
            close_neighbour=False
            for elem in newdiv:
                if max(abs(point[0]-elem[0]),abs(point[1]-elem[1]))<=3:
                    close_neighbour=True
            if not close_neighbour:
                newdiv.append(point)
        
        for i in range(len(newdiv)):
            temp=[]
            for seg in extr_seg:
                if max(abs(seg[-1][0]-newdiv[i][0]),abs(seg[-1][1]-newdiv[i][1]))<=3:
                    temp.append(seg)
                    
            if len(temp)==3:# only possible case for a 3 branches graph only
                maxi=np.argmax(np.array([len(temp[0]),len(temp[1]),len(temp[2])]))
                body=temp.pop(maxi)
                fusi=[]
                if len(temp[0])>alpha*len(temp[1]):
                    return [body+temp[0][-2::-1]]
                elif len(temp[1])>alpha*len(temp[0]):
                    return [body+temp[1][-2::-1]]
                else :
                    debut=min(len(temp[0]),len(temp[1]))
                    for i in range(debut):
                        a=(temp[0][len(temp[0])-debut+i][0]+temp[1][len(temp[1])-debut+i][0])//2
                        b=(temp[0][len(temp[0])-debut+i][1]+temp[1][len(temp[1])-debut+i][1])//2
                        fusi.append([a,b]) 
                        
                    return [body+one_D_line(fusi)[-2::-1]]
            
            elif len(temp)==2:
                if len(temp[0])>alpha*len(temp[1]):
                    new_seg.append(temp[0]) 
                elif len(temp[1])>alpha*len(temp[0]):
                    new_seg.append(temp[1]) 
                else :
                    fusi=[]
                    debut=min(len(temp[0]),len(temp[1]))
                    for i in range(debut):
                        a=(temp[0][len(temp[0])-debut+i][0]+temp[1][len(temp[1])-debut+i][0])//2
                        b=(temp[0][len(temp[0])-debut+i][1]+temp[1][len(temp[1])-debut+i][1])//2
                        fusi.append([a,b]) 
                    new_seg.append(one_D_line(fusi))
            elif len(temp)==1:
                new_seg.append(temp[0])
                
        return new_seg



''' Computing the translation vector between an image and its child and saving it under translation_vector'''
def mask_displacement(dic):
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        child=dic[fichier]['child']
        mask_p=main_mask(dic[fichier]['masks'])
        mask_c=main_mask(dic[child]['masks'])
        
        angle=dic[child]['angle']-dic[fichier]['angle']
        if angle==0:
            #vecguess=dic[fichier]['main_centroid']-dic[child]['main_centroid']
            dic[fichier]['translation_vector']=opt_trans_vec2(mask_p,mask_c)#,rad,vecguess
        else:
            dim1,dim2=np.shape(mask_p)
            centerpoint=np.array([dim1//2,dim2//2],dtype=np.int32)
            #vecguess=(rotation_vector(angle,dic[fichier]['main_centroid'],centerpoint)-dic[child]['main_centroid']).astype(np.int32)
            mask_p=rotation_img(angle,mask_p,centerpoint)
            dic[fichier]['translation_vector']=opt_trans_vec2(mask_p,mask_c).astype(np.int32)#,rad,vecguess
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
    new_mask=np.zeros((l1,l2),dtype=np.int32)
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



def opt_trans_vec2(img_1, img_2):
    corr = scipy.signal.fftconvolve(img_1, img_2[::-1, ::-1])
    argmax = np.unravel_index(corr.argmax(), corr.shape)
    vec = np.array(argmax) - np.array(img_1.shape) + 1
    return vec

#rotation of a vector around a point, the angle is in radian (float as output)
@jit 
def rotation_vector(angle,vec,point):
    mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    newvec=vec-point
    var=np.array([mat[0,0]*newvec[0]+mat[0,1]*newvec[1],mat[1,0]*newvec[0]+mat[1,1]*newvec[1]])
    return point+var

#rotation of an image around a point (int32 as input)
@jit 
def rotation_img(angle,img,point):
    dim1,dim2=np.shape(img)
    new_img=np.zeros((dim1,dim2),dtype=np.int32)
    for i in range(dim1):
        for j in range(dim2):
            trans_vec=rotation_vector(-angle,np.array([i,j]),point)#sign of the rotation : definition of the angle in the logs
            i_n,j_n=trans_vec[0],trans_vec[1]
            i_t=int(i_n)
            j_t=int(j_n)
            if 0<=i_t<dim1-1 and 0<=j_t<dim2-1:
                frac_i=i_n-i_t
                frac_j=j_n-j_t
                new_img[i,j]=np.int32(frac_i*frac_j*img[i_t,j_t]+frac_i*(1-frac_j)*img[i_t,j_t+1]+(1-frac_i)*frac_j*img[i_t+1,j_t]+(1-frac_i)*(1-frac_j)*img[i_t+1,j_t+1])
    return new_img




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
            
            angle=dic[child]['angle']-dic[fichier]['angle']
            
            if angle!=0:            #check on real data
                dim1,dim2=np.shape(mask_p)
                centerpoint=np.array([dim1//2,dim2//2],dtype=np.int32)
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
            
            if angle!=0:            #check on real data sign of angle
                dim1,dim2=np.shape(mask_p)
                centerpoint=np.array([dim1//2,dim2//2],dtype=np.int32)
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
    result_p=np.zeros(number_mask_p,dtype=np.int32)
    result_c=np.zeros(number_mask_c,dtype=np.int32)
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
    
    dic[fichier]['basic_graph']=[]
    
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
def plot_graph_masks(dic,graph_name,maskcol,saving_video=False,video_path=''):
    #Initialisation
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        # plot image with masks overlaid
        img = cv2.imread(dic[fichier]['adress'],0)
        masks=dic[fichier][graph_name+'_masks']
        masknumber=np.max(masks)
        colormask=copy.deepcopy(maskcol)
        numbercolor=len(colormask)
        colormask=(masknumber//numbercolor+1)*colormask
        colormask=np.array(colormask[:masknumber])
        mask_RGB = plot.mask_overlay(img,masks,colors=colormask)#image with masks
        plt.title('time : '+str(dic[fichier]['time']))
        plt.imshow(mask_RGB)
        
        # plot the centroids and the centerlines
        centr=dic[fichier]['centroid']
        line=dic[fichier]['centerlines']
        for i in range(len(centr)):
            #centroids
            plt.plot(centr[i,1], centr[i,0], color='k',marker='o')
            plt.annotate(str(int(dic[fichier][graph_name+'_graph_values'][i])), centr[i,::-1], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
            #centerlines
            if len(line[i])>1:
                plt.plot(line[i][:,1],line[i][:,0], color='k')
            
        main_centroid=dic[fichier]['main_centroid']
        plt.plot(main_centroid[1], main_centroid[0], color='w',marker='o')
            
        #plot the displacement of the centroid between two images
        next_centr=dic[dic[fichier]['child']]['centroid']
        for link in dic[fichier][graph_name+'_graph']:
            if next_centr[link[1]-1][1]!=centr[link[0]-1][1] or next_centr[link[1]-1][0]!=centr[link[0]-1][0]:
                plt.annotate("", xy=(next_centr[link[1]-1][1], next_centr[link[1]-1][0]), xycoords='data', xytext=(centr[link[0]-1][1], centr[link[0]-1][0]), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='w'))
            else :
                plt.plot(next_centr[link[1]-1][1], next_centr[link[1]-1][0],color='w',marker='o', markersize=0.5)
        if saving_video:
            plt.savefig(video_path+'Img'+fichier+'.png', format='png',bbox_inches='tight',dpi=500)
        
        plt.show()
        fichier=dic[fichier]['child']
    
    # ploting last image
    img = cv2.imread(dic[fichier]['adress'],0)
    masks=dic[fichier][graph_name+'_masks']
    masknumber=np.max(masks)
    colormask=copy.deepcopy(maskcol)
    numbercolor=len(colormask)
    colormask=(masknumber//numbercolor+1)*colormask
    colormask=np.array(colormask[:masknumber])
    mask_RGB = plot.mask_overlay(img,masks,colors=colormask)#image with masks
    plt.title('time : '+str(dic[fichier]['time']))
    plt.imshow(mask_RGB)
    
    
    # plot the centroids and centerlines
    centr=dic[fichier]['centroid']
    line=dic[fichier]['centerlines']
    for i in range(len(centr)):
        #centroids
        plt.plot(centr[i,1], centr[i,0], color='k',marker='o')
        plt.annotate(str(int(dic[fichier][graph_name+'_graph_values'][i])), centr[i,::-1], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
        #centerlines
        if len(line[i])>1:
            plt.plot(line[i][:,1],line[i][:,0], color='k')
    main_centroid=dic[fichier]['main_centroid']
    plt.plot(main_centroid[1], main_centroid[0], color='w',marker='o')
    plt.show()



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




''' Final function to run the whole pipeline on a dataset'''

def run_one_dataset(Main_directory,dir_property="Height/",cropped=False, cel_gpu=True):
    print(Main_directory)
    
    dir_name=Main_directory+dir_property            #name of dir
    my_data = "../data/"+dir_name       #path of dir
    #directory with the usefull information of the logs, None if there is no.
    data_log="../data/"+Main_directory+"log/"             #       #path and name

    #Temporary directories
    #directory of cropped data each image of the dataset is cropped to erase the white frame
    cropped_data=dir_name+"cropped_data/" 
    #directory with the usefull info extracted from the logs
    cropped_log=dir_name+"log_cropped_data/" 
    #directory for output data from cellpose 
    segments_path = dir_name+"cellpose_output/"


    #Outputs
    #directory of the processed images (every image has the same pixel size and same zoom)
    final_data=dir_name+"final_data/" 
    #dictionnary and dimension directory
    Dic_dir=dir_name+"Dic_dir/" 
    #Saving path for the dictionnary
    saving_path=Dic_dir+'Main_dictionnary'
    #dimension and scale of all the final data
    dimension_data=Dic_dir+'Dimension'
    #dimension and scale of all the final data
    list_savingpath=Dic_dir+'masks_list'
    


    
    ''' Parameters'''
    #cropping parameters
    if cropped:
        crop_up=27
        crop_down=1
        crop_left=30
        crop_right=101
    else:
        crop_up=1
        crop_down=1
        crop_left=1
        crop_right=1
    
    
    
    
    #cellpose parameters
    
    cel_model_type='cyto'
    cel_channels=[0,0]  # define CHANNELS to run segementation on grayscale=0, R=1, G=2, B=3; channels = [cytoplasm, nucleus]; nucleus=0 if no nucleus
    cel_diameter_param = 1 # parameter to adjust the expected size (in pixels) of bacteria. Incease if cellpose detects too small masks, decrease if it don't detects small mask/ fusion them. Should be around 1 
    cel_flow_threshold = 0.9 
    cel_cellprob_threshold=0.0
    batch_size=8
    
    
    #erasing small masks that have a smaller relative size :
    ratio_erasing_masks=0.2
    #erasing masks that have a ratio of saturated surface bigger than :
    ratio_saturation=0.1
    
    
    #fraction of the preserved area to consider child and parent relation for masks
    surface_thresh=0.6
    
    #minimum ratio of two brench length to erase the small branch
    centerline_crop_param=2
    
    #colors of the masks
    color_mask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]

    

    data_prep(my_data,crop_up,crop_down,crop_left,crop_right,cropped_data,cropped_log,my_data,log_dir=data_log)
    
    print("data_prep",0)

    dimension_def(cropped_data,cropped_log,final_data,dimension_data,Dic_dir)

    print("dimension_def",1)

    run_cel(final_data,cel_gpu,cel_model_type,cel_channels,cel_diameter_param,cel_flow_threshold,cel_cellprob_threshold,segments_path,dimension_data,batch_size=batch_size)

    print("run_cel",2)
    
    main_dict=download_dict(final_data,cropped_log,segments_path,dimension_data)

    print("download_dict",3)

    main_parenting(main_dict)
    
    print("main_parenting",4)

    centroid_area(main_dict)

    print("centroid_area",5)

    clean_masks(ratio_erasing_masks,ratio_saturation, main_dict)
    
    print("clean_masks",6)
    
    main_parenting(main_dict) #re-run in case all masks in a frame are erased

    print("main_parenting",7)
    
    construction_mask_list(main_dict,Main_directory,list_savingpath)
    
    print("construction_mask_list",8)

    centerline_mask(main_dict,centerline_crop_param,saving=True,savingpath=saving_path)
    
    print("centerline_mask",9)
    '''
    mask_displacement(main_dict)

    print("mask_displacement",10)

    relation_mask(main_dict,surface_thresh)

    print("relation_mask",11)

    basic_graph(main_dict,saving=True,savingpath=saving_path)

    print("basic_graph",12)
    plot_graph(main_dict,'basic',color_mask)
    plot_graph_masks(main_dict,'basic',color_mask)
    '''



if __name__ == "__main__":
    
    '''Running the different functions'''
    ''''''
    run_one_dataset(Directory,cropped=True)
    
    
    
    # for elem in data_set:
    #     if len(elem)==2:
    #         run_one_dataset(elem[0],cropped=elem[1])
    #     elif len(elem)==3:
    #         run_one_dataset(elem[0],cropped=elem[1],cel_gpu=elem[2])
        
        
        
    # color_mask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]

    # path=Directory+result_path+dic_name
    # list_path=Directory+result_path+'masks_list'+'.npy'
    # masks_list=np.load(list_path, allow_pickle=True)
    # main_dict=np.load(path, allow_pickle=True).item()
    # plot_graph(main_dict,'basic',color_mask)
    # plot_graph_masks(main_dict,'basic',color_mask,saving_video=False,video_path='Video/')