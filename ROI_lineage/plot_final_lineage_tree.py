#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:34:34 2023

@author: c.soubrier
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot
import cv2
import processing as pr
import extract_individuals as exi

Directory= 'delta_lamA_03-08-2018/2/'# "dataset/"

result_path='Height/Dic_dir/'

dic_name='Main_dictionnary.npy'

list_name='masks_list.npy'

ROI_dictionary='ROI_dict.npy'

final_thresh=0.75

thres_min_division=0.6 #threshold to consider that there is a division and not just a break in the ROI

lin_mat_name='non_trig_Link_matrix.npy'

Bool_mat_name="Bool_matrix.npy"

index_list_name='masks_ROI_list.npy'

min_len_ROI=3

data_set=[["dataset/",True],["delta_3187/21-02-2019/",True],["delta_3187/19-02-2019/",True],["delta_parB/03-02-2015/",False],["delta_parB/15-11-2014/",False],["delta_parB/18-01-2015/",False],["delta_parB/18-11-2014/",False],["delta_lamA_03-08-2018/1/",True],["delta_lamA_03-08-2018/2/",True],["WT_mc2_55/06-10-2015/",False],["WT_mc2_55/05-10-2015/",False],["WT_mc2_55/30-03-2015/",True],["WT_mc2_55/05-02-2014/",False],["WT_11-02-15/",False,False],["delta_ripA/14-10-2016/",False],["delta_ripA/160330_rip_A_no_inducer/",True],["delta_ripA/160407_ripA_stiffness_septum/",True],["delta_LTD6_04-06-2017/",False]  ]


colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]

ROI_name='ROI_dict.npy' 

#Dictionnary of the ROIs with the ROI name as entry and as argument : Parent ROI (string, if not ''), Child1 ROI (string, if not ''), Child2 ROI (string, if not ''), list of masks (int), ROI index, Roi color_index


#Update the main_dic with the ROI of each mask






def construct_ROI(dirname,resultpath,bool_name,saving=True,ROIname=ROI_name):
    
    bool_mat=np.load(dirname+result_path+bool_name)
    ROI_dict = {} # initialize dictionary

    roots = exi.get_roots(bool_mat) # get all roots and start new tree traversal for each one 
    starting_num = 1 # initial ROI ID number is 1

    for root in roots:
        individuals = exi.get_individuals(bool_mat,root,ROI_num=starting_num,root=root)[0]
        ROI_dict.update(exi.list2dict(individuals))
        starting_num = int(list(ROI_dict.keys())[-1].split()[-1])+1 # make sure next tree's first ROI is latest ROI ID# + 1

    create_children(ROI_dict) # add Children subkey to each dictionary item

    #print(ROI_dict)
    if saving:
        np.save(dirname+resultpath + ROIname,ROI_dict)
    return ROI_dict




def filter_good_ROI_dic(ROI_dic,min_number):
    newdic={}
    for ROI in ROI_dic.keys():
        if len(ROI_dic[ROI]['Mask IDs'])>=min_number:
            newdic[ROI]=ROI_dic[ROI]
    termination=False
    while not termination:
        termination=True
        keys=list(newdic.keys())
        for ROI in keys:
            parent=ROI_dic[ROI]['Parent']
            if parent!='' and parent not in keys:
                termination=False
                newdic[parent]=ROI_dic[parent]
    termination=False
    while not termination:
        termination=True
        keys=list(newdic.keys())
        for ROI in keys:
            children=ROI_dic[ROI]['Children']
            if children!=[] and len(children)<=2:
                for i in [0,1]:
                    if children[i] not in keys:
                        termination=False
                        newdic[children[i]]=ROI_dic[children[i]]
            elif len(children)>2:
                print('More then 2 children '+ ROI, ROI_dic[ROI]['Children'])
                for child in ROI_dic[ROI]['Children']:
                    if child in newdic.keys():
                        newdic[child]['Parent']=''
                ROI_dic[ROI]['Children']=[]

    return newdic

def ROI_index(ROI_dic):
    count=1
    colorcount=1
    for ROI in ROI_dic.keys():
        ROI_dic[ROI]['color_index']=colorcount
        colorcount+=1
        if ROI_dic[ROI]['Parent']=='':
          ROI_dic[ROI]['index']=str(count)+'/'
          count+=1
        else:
            ROI_dic[ROI]['index']=''
    for ROI in ROI_dic.keys():
        update_ROI_index(ROI,ROI_dic)
        
def create_children(ROI_dic):
    for ROI in ROI_dic.keys():
        ROI_dic[ROI]['Children']=[]
    for ROI in ROI_dic.keys():
        if ROI_dic[ROI]['Parent']!='':
          ROI_dic[ROI_dic[ROI]['Parent']]['Children'].append(ROI)
              


def update_ROI_index(ROI,ROI_dic,terminal=True):
    if ROI_dic[ROI]['index']=='':
        parent=ROI_dic[ROI]['Parent']
        index_parent=update_ROI_index(parent,ROI_dic,terminal=False)
        if ROI==ROI_dic[parent]['Children'][0]:
            new_index=index_parent+'0'
        else:
            new_index=index_parent+'1'
        ROI_dic[ROI]['index']=new_index
        if not terminal:
            return new_index
    elif not terminal:
        return ROI_dic[ROI]['index']
    
    
    
def intensity_lineage(index):
    res=re.findall(r'\d+\.\d+|\d+',index)
    if len(res)==1:
        return int(res[0])
    else:
        num,suc=res[0],res[1]
        count=int(num)
        for i in range(len(suc)):
            count+= 2**(-i-1)*(1/2-int(suc[i]))
        return count




def plot_lineage_tree(ROI_dic,masks_list,main_dic,maskcol):
    
    for ROI in ROI_dic.keys():
        if ROI_dic[ROI]['Parent']!='':
            parent=ROI_dic[ROI]['Parent']
            value1=intensity_lineage(ROI_dic[parent]['index'])
            value2=intensity_lineage(ROI_dic[ROI]['index'])
            point1=ROI_dic[parent]['Mask IDs'][-1]
            point2=ROI_dic[ROI]['Mask IDs'][0]
            t1=main_dic[masks_list[point1][2]]['time']
            t2=main_dic[masks_list[point2][2]]['time']
            plt.plot([t1,t2],[value1,value2],color='k')
            value1,t1=value2,t2
            color=ROI_dic[ROI]['color_index']
        else:
            value1=intensity_lineage(ROI_dic[ROI]['index'])
            point1=ROI_dic[ROI]['Mask IDs'][0]
            t1=main_dic[masks_list[point1][2]]['time']
            color=ROI_dic[ROI]['color_index']
        
        point2=ROI_dic[ROI]['Mask IDs'][-1]
        t2=main_dic[masks_list[point2][2]]['time']
        len_col=len(maskcol)
        col=maskcol[int(color%len_col)]
        
        plot_col=(col[0]/255,col[1]/255,col[2]/255)
        plt.plot([t1,t2],[value1,value1],color=plot_col)
        
    plt.show()
    
def extract_roi_list_from_dic(ROI_dic,masks_list):
    newlist=np.zeros((len(masks_list),3),dtype=object)
    for ROI in ROI_dic.keys():
        color=ROI_dic[ROI]['color_index']
        index=ROI_dic[ROI]['index']
        for i in ROI_dic[ROI]['Mask IDs']:
            newlist[i,0]=color
            newlist[i,1]=index
            newlist[i,2]=ROI
    return newlist




def plot_image_lineage_tree(ROI_dic,masks_list,dic,maskcol):
    indexlist=extract_roi_list_from_dic(ROI_dic,masks_list)
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        # plot image with masks overlaid
        img = cv2.imread(dic[fichier]['adress'],0)
        masks=dic[fichier]['masks']
        masknumber=np.max(masks)
        col_ind_list=np.zeros(masknumber,dtype=np.int32)
        roi_ind_list=[]
        for i in range(masknumber):
            roi_ind_list.append([])
        for i in range(masknumber):
            elem=indexlist[dic[fichier]["mask_list"][i]]
            col_ind_list[i]=elem[0]
            roi_ind_list[i]=str(elem[1])
        
        
        len_col=len(maskcol)
        for i in range(masknumber):
            if roi_ind_list[i]=='0':
                col_ind_list[i]=0
            else:
                col_ind_list[i]=col_ind_list[i]%len_col+1
        masks=pr.update_masks(masks,col_ind_list)
        colormask=np.array(maskcol)
        mask_RGB = plot.mask_overlay(img,masks,colors=colormask)#image with masks
        plt.title('time : '+str(dic[fichier]['time']))
        plt.imshow(mask_RGB)
        
        # plot the centroids and the centerlines
        centr=dic[fichier]['centroid']
        line=dic[fichier]['centerlines']
        for i in range(len(centr)):
            #centroids
            plt.plot(centr[i,1], centr[i,0], color='k',marker='o')
            plt.annotate(roi_ind_list[i], centr[i,::-1], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
            if len(line[i])>1:
                plt.plot(line[i][:,1],line[i][:,0], color='k')
        
        main_centroid=dic[fichier]['main_centroid']
        plt.plot(main_centroid[1], main_centroid[0], color='w',marker='o')
            
        #plot the displacement of the centroid between two images
        plt.show()
        fichier=dic[fichier]['child']
    
    # plot image with masks overlaid
    img = cv2.imread(dic[fichier]['adress'],0)
    masks=dic[fichier]['masks']
    masknumber=np.max(masks)
    col_ind_list=np.zeros(masknumber,dtype=np.int32)
    roi_ind_list=[]
    for i in range(masknumber):
        roi_ind_list.append([])
    for i in range(masknumber):
        elem=indexlist[dic[fichier]["mask_list"][i]]
        col_ind_list[i]=elem[0]
        roi_ind_list[i]=str(elem[1])
    
    
    len_col=len(maskcol)
    for i in range(masknumber):
        if roi_ind_list[i]=='0':
            col_ind_list[i]=0
        else:
            col_ind_list[i]=col_ind_list[i]%len_col+1
    masks=pr.update_masks(masks,col_ind_list)
    colormask=np.array(maskcol)
    mask_RGB = plot.mask_overlay(img,masks,colors=colormask)#image with masks
    plt.title('time : '+str(dic[fichier]['time']))
    plt.imshow(mask_RGB)
    
    # plot the centroids and the centerlines
    centr=dic[fichier]['centroid']
    line=dic[fichier]['centerlines']
    for i in range(len(centr)):
        #centroids
        plt.plot(centr[i,1], centr[i,0], color='k',marker='o')
        plt.annotate(roi_ind_list[i], centr[i,::-1], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
        if len(line[i])>1:
            plt.plot(line[i][:,1],line[i][:,0], color='k')
    
    main_centroid=dic[fichier]['main_centroid']
    plt.plot(main_centroid[1], main_centroid[0], color='w',marker='o')
        
    #plot the displacement of the centroid between two images
    plt.show()



    
def rank_subtrees(ROI_dic,ROI_min_number):
    max_root=0
    for ROI in ROI_dic.keys():
        root_index=round(ROI_dic[ROI]['index'])
        if root_index>max_root:
            max_root=root_index
    rank=np.zeros(max_root)
    for ROI in ROI_dic.keys():
        root_index=round(ROI_dic[ROI]['index'])
        rank[root_index-1]+=1
    order=np.argsort(rank)[::-1]
    stop_number=-1
    for i in range(len(order)):
        if root_index[order[i]]>=ROI_min_number:
            order[i]+=1
        else:
            stop_number=i
            break
    return order[:stop_number]
    


def detect_bad_div(ROI_dic,linmatrix,masks_list,thres,thres_min):
    indexlist=extract_roi_list_from_dic(ROI_dic,masks_list)
    for ROI in list(ROI_dic.keys()):
        if ROI_dic[ROI]['Parent']=='':
            first_elem=ROI_dic[ROI]['Mask IDs'][0]
            elem=first_elem
            termination=False
            while not termination and elem>0:
                elem-=1
                if linmatrix[first_elem,elem]>thres and linmatrix[elem,first_elem]<thres_min and indexlist[elem][1]!='0' and indexlist[elem][1]!=0:
                    
                    if ROI_dic[indexlist[elem][2]]['Mask IDs'][-1]<first_elem:
                    
                        print('regluing' +ROI,elem)
                        
                        if ROI_dic[indexlist[elem][2]]['Children']==[]:
                            newindex=indexlist[elem][1]+'0'
                            
                            ROI_dic[ROI]['Parent']=indexlist[elem][2]
                            ROI_dic[indexlist[elem][2]]['Children'].append(ROI)
                            
                            change_root_index(ROI,ROI_dic,newindex,indexlist)
                            
                        elif len(ROI_dic[indexlist[elem][2]]['Children'])==1:
                            newindex=indexlist[elem][1]+'1'
                            
                            ROI_dic[indexlist[elem][2]]['Children'].append(ROI)
                            ROI_dic[ROI]['Parent']=indexlist[elem][2]
                            
                            change_root_index(ROI,ROI_dic,newindex,indexlist)
                            
                        else:
                            print('Error in re-gluing ROI',ROI,ROI_dic[indexlist[elem][2]])
                        termination=True
    return indexlist






def change_root_index(ROI,ROI_dic,newindex,indexlist):
    var=re.findall(r'\d+\.\d+|\d+',ROI_dic[ROI]['index'])
    if len(var)==1:
        index=newindex
    else:
        index=newindex+var[1]
    ROI_dic[ROI]['index']=index
    for elem in ROI_dic[ROI]['Mask IDs']:
        indexlist[elem][1]=index
    for child in ROI_dic[ROI]['Children']:
        change_root_index(child,ROI_dic,newindex,indexlist)

          
                
    
    
    
    
    
def run_whole_lineage_tree(direc,resultpath,dicname,listname,ROIdict,indexlistname,maskcol,linmatname,boolmatname,thres,min_number,thresmin):
    
    masks_list=np.load(direc+resultpath+listname, allow_pickle=True)
    main_dict=np.load(direc+resultpath+dicname, allow_pickle=True).item()
    ROI_dict=np.load(direc+resultpath+ROIdict,allow_pickle=True).item()
    
    ROI_dict=construct_ROI(direc,resultpath,boolmatname)
    #print(ROI_dict)
    print(ROI_dict['ROI 521'],ROI_dict['ROI 522'],ROI_dict['ROI 523'],ROI_dict['ROI 524'])
    linmatrix=np.load(direc+resultpath+linmatname)
    
    newdic=filter_good_ROI_dic(ROI_dict,min_number)
    
    ROI_index(newdic)
    
    indexlist=detect_bad_div(newdic,linmatrix,masks_list,thres,thresmin)
    #print(newdic)
    plot_lineage_tree(newdic,masks_list,main_dict,maskcol)
    plot_image_lineage_tree(newdic,masks_list,main_dict,maskcol)
    np.save(direc+resultpath+ROIdict,newdic,allow_pickle=True)
    np.save(direc+resultpath+indexlistname,indexlist)
    #return newdic,indexlist
    
    

if __name__ == "__main__":
    run_whole_lineage_tree(Directory,result_path,dic_name,list_name,ROI_dictionary,index_list_name,colormask,lin_mat_name,Bool_mat_name,final_thresh,min_len_ROI,thres_min_division)
    # for direc in data_set:
    #     print(direc[0])
    #     run_whole_lineage_tree(direc[0],result_path,dic_name,list_name,ROI_dictionary,index_list_name,colormask,lin_mat_name,Bool_mat_name,final_thresh,min_len_ROI)


    