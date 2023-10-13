#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:34:34 2023

@author: c.soubrier
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot
from numba import njit
import extract_individuals as exi


Directory="WT_mc2_55/30-03-2015/"#"dataset/"# 'delta_lamA_03-08-2018/2/'

data_set=['delta_lamA_03-08-2018/','delta_LTD6_04-06-2017/',"delta_parB/03-02-2015/","delta_parB/15-11-2014/","delta_parB/18-01-2015/","delta_parB/18-11-2014/","delta_ripA/14-10-2016/","WT_mc2_55/06-10-2015/","WT_mc2_55/30-03-2015/","WT_mc2_55/03-09-2014/",'WT_INH_700min_2014/','WT_CCCP_irrigation_2016/','WT_filamentation_cipro_2015/']



''' Parameters '''

#minimal threshold to consider a daughter-mother relation
final_thresh=0.75 #0.8

#threshold to consider that there is a division and not just a break in the ROI
thres_min_division=0.7 #0.7 

#threshold to consider that two children are actually linked
comparison_thres=0.7
#minimum number of frames in an ROI
min_len_ROI=3 


''' Output'''

#Dictionnary of the ROIs with the ROI name as entry and as argument : Parent ROI (string, if not ''), Child1 ROI (string, if not ''), Child2 ROI (string, if not ''), list of masks (int), ROI index, Roi color_index



''' Functions'''

def filter_good_ROI_dic(ROI_dic,min_number):
    newdic={}
    for ROI in ROI_dic.keys(): #filtering the ROI with a minimum number of frames
        if len(ROI_dic[ROI]['Mask IDs'])>=min_number:
            newdic[ROI]=ROI_dic[ROI]
    termination=False
    while not termination: #updating the ancestors of the good ROI 
        termination=True
        keys=list(newdic.keys())
        for ROI in keys:
            parent=ROI_dic[ROI]['Parent']
            if parent!='' and parent not in keys:
                termination=False
                newdic[parent]=ROI_dic[parent]
    termination=False
    while not termination: #updating the descendants of the good ROI 
        termination=True
        keys=list(newdic.keys())
        for ROI in keys:
            children=ROI_dic[ROI]['Children']
            
            
            # if children!=[]:
            #     for elem in children:
            #         if elem not in keys:
            #             termination=False
            #             newdic[elem]=ROI_dic[elem]
            
            
            if children!=[] and len(children)<=2:
                for i in [0,1]:
                    if children[i] not in keys:
                        termination=False
                        newdic[children[i]]=ROI_dic[children[i]]
            elif len(children)>2:
                for child in ROI_dic[ROI]['Children']:
                    if child in newdic.keys():
                        newdic[child]['Parent']=''
                ROI_dic[ROI]['Children']=[]

    return newdic


def check_division(ROI_dic,linmat, masklist,maindic,comp_thres):
    keys=list(ROI_dic.keys())[::-1]
    for ROI in keys:
        if ROI in ROI_dic.keys():
            children=ROI_dic[ROI]['Children']
            
            if len(children)==2: #checking if the masks of the two children are intersecting (wrong division) and deleting the wrong branch
                score = max([linmat[i,j] for i in ROI_dic[children[0]]['Mask IDs'] for j in ROI_dic[children[1]]['Mask IDs']]) 
                if score>=comp_thres:
                    print(ROI,children)
                    #selecting the ROI with the longest life
                    t0=latest_descendant_time(children[0],ROI_dic,masklist,maindic)
                    t1=latest_descendant_time(children[1],ROI_dic,masklist,maindic)
    
                    chosen=np.argmax(np.array([t0,t1]))
                    nonchosen=1-chosen
                    #updating the dictionnary
                    
                    ROI_dic[ROI]['Children']=ROI_dic[children[chosen]]['Children']
                    ROI_dic[ROI]['Mask IDs']+=ROI_dic[children[chosen]]['Mask IDs']
                    for elem in ROI_dic[children[chosen]]['Children']:
                        ROI_dic[elem]['Parent']=ROI
                    for elem in ROI_dic[children[nonchosen]]['Children']:
                        ROI_dic[elem]['Parent']=''
                    del(ROI_dic[children[0]])
                    del(ROI_dic[children[1]])
                    
            elif len(children)==3:
                score02 = max([linmat[i,j] for i in ROI_dic[children[0]]['Mask IDs'] for j in ROI_dic[children[2]]['Mask IDs']])
                score12 = max([linmat[i,j] for i in ROI_dic[children[1]]['Mask IDs'] for j in ROI_dic[children[2]]['Mask IDs']])
                score01 = max([linmat[i,j] for i in ROI_dic[children[0]]['Mask IDs'] for j in ROI_dic[children[1]]['Mask IDs']])
                
                probnum = score01>=comp_thres + score02>=comp_thres + score12>=comp_thres
                
                
                
                if probnum==3:
                    ROI_dic[ROI]['Children']=[]
                    for elem in children:
                        for subelem in ROI_dic[elem]['Children']:
                            ROI_dic[subelem]['Parent']=''
                        del ROI_dic[elem]
                        
                        
                        
                elif probnum==2:
                    nonchosen=0*(score12<comp_thres)+1*(score02<comp_thres)+2*(score02<comp_thres)
                    for elem in ROI_dic[children[nonchosen]]['Children']:
                        ROI_dic[elem]['Parent']=''
                    
                    del ROI_dic[children[nonchosen]]
                    del(ROI_dic[ROI]['Children'][nonchosen])    
                
                
                elif probnum==1:
                    if score01>=comp_thres:
                        elem0=0
                        elem1=1
                    elif score02>=comp_thres :
                        elem0=0
                        elem1=2
                    else :
                        elem0=1
                        elem1=2
                    #selecting the ROI with the longest life
                    t0=latest_descendant_time(children[elem0],ROI_dic,masklist,maindic)
                    t1=latest_descendant_time(children[elem1],ROI_dic,masklist,maindic)
    
                    if np.argmin(np.array([t0,t1])):
                        nonchosen=elem1
                    else:
                        nonchosen=elem0
                    #updating the dictionnary
                    
                    for elem in ROI_dic[children[nonchosen]]['Children']:
                        ROI_dic[elem]['Parent']=''
                    
                    del ROI_dic[children[nonchosen]]
                    del(ROI_dic[ROI]['Children'][nonchosen])    
                        
                else:
                    t0=latest_descendant_time(children[0],ROI_dic,masklist,maindic)
                    t1=latest_descendant_time(children[1],ROI_dic,masklist,maindic)
                    t2=latest_descendant_time(children[2],ROI_dic,masklist,maindic)
                    chosen=np.argmax(np.array([t0,t1,t2]))
                    
                    for child in ROI_dic[children[chosen]]['Children']:
                        ROI_dic[child]['Parent']=''
                    del(ROI_dic[ROI]['Children'][chosen])
                    del(ROI_dic[children[chosen]])
                    
                    
            elif len(children)>3:
                print('Too many children '+ ROI, ROI_dic[ROI]['Children'])
                for child in children:
                    ROI_dic[child]['Parent']=''
                    if ROI_dic[child]['Children']==[]:
                        del ROI_dic[child]
                ROI_dic[ROI]['Children']=[]
                
def latest_descendant_time(ROI,ROI_dic,masklist,maindic):
    if ROI_dic[ROI]['Children']==[]:
        index=ROI_dic[ROI]['Mask IDs'][-1]
        return maindic[masklist[index][2]]['time']
    else:
        return max([latest_descendant_time(children,ROI_dic,masklist,maindic) for children in ROI_dic[ROI]['Children']])
    

def ROI_index(ROI_dic,masklist,maindic):
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
        list_good_qual=np.ones(len(ROI_dic[ROI]['Mask IDs']))
        for i in range(len(ROI_dic[ROI]['Mask IDs'])):
            list_good_qual[i]=isgoodmask(ROI_dic[ROI]['Mask IDs'][i],masklist,maindic)
        ROI_dic[ROI]['masks_quality']=list_good_qual
    for ROI in ROI_dic.keys():
        
        update_ROI_index(ROI,ROI_dic)
        
def isgoodmask(number,masklist,maindic,dist=0.6): #if a mask is at a distance less than dist (in mu m) of the frame 
    frame=masklist[number][2]
    mask_num=masklist[number][3]
    masks=maindic[frame]['masks']==mask_num
    pixel_number=int(np.ceil(dist/maindic[frame]['resolution']))
    return(not (np.max(masks[:pixel_number,:]) or np.max(masks[-pixel_number-1:,:]) or np.max(masks[:,:pixel_number]) or np.max(masks[:,-pixel_number-1:])))



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




def plot_lineage_tree(ROI_dic,masks_list,main_dic,maskcol,directory):
    plt.figure()
    
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
        
        # plt.annotate(ROI, ((t2+t1)/2,value1))
        plt.title('Lineage tree '+directory)
    plt.xlabel('time (mn)')
    plt.ylabel('Generation index')
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




def plot_image_one_ROI(ROI,ROI_dic,masks_list,dic):
    plotlist=ROI_dic[ROI]['Mask IDs']
    Roi_index=ROI_dic[ROI]['index']
    for maskindex in plotlist:
        file=masks_list[maskindex][2]
        index=masks_list[maskindex][3]
        img = np.load(dic[file]['adress'])['Height_fwd']
        mask=(dic[file]['masks']==index).astype(int)
        mask_RGB =plot.mask_overlay(img,mask)
        plt.figure()
        plt.title('time : '+str(dic[file]['time']))
        plt.imshow(mask_RGB)
        centroid=dic[file]['centroid'][index-1]
        center=dic[file]['centerlines'][index-1]
        plt.annotate(Roi_index, centroid[::-1], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
        if len(center)>1:
            plt.plot(center[:,1],center[:,0], color='k')
        plt.show()
        


@njit    
def renorm_img(img):
    dim1,dim2=np.shape(img)
    newimg=np.zeros((dim1,dim2))
    maxi=np.max(img)
    mini=np.min(img)
    if maxi==mini:
        return newimg
    else:
        for i in range(dim1):
            for j in range(dim2):
                newimg[i,j]=(img[i,j]-mini)/(maxi-mini)
        return newimg


@njit
def update_masks(mask,new_values):
    (l,L)=np.shape(mask)
    for j in range(l):
        for k in range(L):
            if mask[j,k]!=0:
                mask[j,k]=new_values[mask[j,k]-1]
    return mask


def plot_image_lineage_tree(ROI_dic,masks_list,dic,maskcol,indexlist,directory):
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        # plot image with masks overlaid
        img = np.load('../data/datasets/'+dic[fichier]['adress'])['Height_fwd']
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
        masks=update_masks(masks,col_ind_list)
        
        colormask=np.array(maskcol)
        mask_RGB = plot.mask_overlay(renorm_img(img),masks,colors=colormask)#image with masks
        plt.figure()
        plt.title(directory+', time : '+str(dic[fichier]['time']))
        plt.imshow(mask_RGB)
        
        # plot the centroids and the centerlines
        centr=dic[fichier]['centroid']
        line=dic[fichier]['centerlines']
        for i in range(len(centr)):
            #centroids
            plt.plot(centr[i,1], centr[i,0], color='k',marker='o')
            plt.annotate(roi_ind_list[i], centr[i,::-1], xytext=[10,0], textcoords='offset pixels', color='lime')
            if len(line[i])>1:
                plt.plot(line[i][:,1],line[i][:,0], color='k')
        
        main_centroid=dic[fichier]['main_centroid']
        plt.plot(main_centroid[1], main_centroid[0], color='w',marker='o')
        plt.savefig('../../Python_code/img/'+fichier+'.png', format='png')
        #plot the displacement of the centroid between two images
        plt.show()
        fichier=dic[fichier]['child']
    
    # plot image with masks overlaid
    img = np.load('../data/datasets/'+dic[fichier]['adress'])['Height_fwd']
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
    masks=update_masks(masks,col_ind_list)
    colormask=np.array(maskcol)
    mask_RGB = plot.mask_overlay(img,masks,colors=colormask)#image with masks
    plt.figure()
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
    

#%% regluing the ROI without a parent to their parent ROI (undetected divisions)
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
                    
                        print('regluing' +ROI,elem, 'index', ROI_dic[ROI]['index'], ROI_dic[indexlist[elem][2]]['index'])
                        
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
                            print('Error in re-gluing ',ROI,ROI_dic[ROI]['index'],indexlist[elem][2],ROI_dic[indexlist[elem][2]]['index'])
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


def manually_regluing(direc,ROIdict,indexlistname,parent,child,division=True):
    ROI_dict=np.load(direc+ROIdict,allow_pickle=True)['arr_0'].item()
    indexlist=np.load(direc+indexlistname,allow_pickle=True)['arr_0']
    parentROI=''
    childROI=''
    for ROI in ROI_dict.keys():
        if ROI_dict[ROI]['index']==parent:
            parentROI=ROI
        elif ROI_dict[ROI]['index']==child:
            childROI=ROI
            
    if parentROI!=''and childROI!='':
        if division:
            if ROI_dict[parentROI]['Children']==[]:
                ROI_dict[parentROI]['Children'].append(childROI)
                ROI_dict[childROI]['Parent']=parentROI
                change_root_index(childROI,ROI_dict,parent+'0',indexlist)
            elif len(ROI_dict[parentROI]['Children'])==1:
                ROI_dict[parentROI]['Children'].append(childROI)
                ROI_dict[childROI]['Parent']=parentROI
                change_root_index(childROI,ROI_dict,parent+'1',indexlist)
            else:
                raise ValueError('already 2 children for '+parent)
        else:
            if not ROI_dict[parentROI]['Children']==[]:
                raise ValueError('already children for '+parent)
            else :
                change_root_index(childROI,ROI_dict,parent,indexlist)
                color=ROI_dict[parentROI]['color_index']
                index=ROI_dict[parentROI]['index']
                for grand_children in ROI_dict[childROI]['Children']:
                    ROI_dict[grand_children]['Parent']=parentROI
                    ROI_dict[parentROI]['Children'].append(grand_children)
                for mask in ROI_dict[childROI]['Mask IDs']:
                    ROI_dict[parentROI]['Mask IDs'].append(mask)
                    indexlist[mask,0]=color
                    indexlist[mask,1]=index
                    indexlist[mask,2]=parentROI
    else:
        raise NameError('No ROI with following index :'+(parentROI=='')*parent+' '+(childROI=='')*child)
    np.savez_compressed(direc+ROIdict,ROI_dict,allow_pickle=True)
    np.savez_compressed(direc+indexlistname,indexlist,allow_pickle=True)
            
                
    
    
    
    
    
def run_whole_lineage_tree(direc,thres=final_thresh,min_number=min_len_ROI,thresmin=thres_min_division,comp_threshold=comparison_thres):
    
    
    dicname='Main_dictionnary.npz'

    listname='masks_list.npz'

    ROIdict='ROI_dict.npz'

    linmatname='non_trig_Link_matrix.npy'

    boolmatname="Bool_matrix.npy"

    linkmatname='Link_matrix.npy'

    indexlistname='masks_ROI_list.npz'
    
    
    colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]


    
    masks_list=np.load('../data/datasets/'+direc+listname, allow_pickle=True)['arr_0']
    main_dict=np.load('../data/datasets/'+direc+dicname, allow_pickle=True)['arr_0'].item()
    Bool_matrix=np.load('../data/datasets/'+direc+boolmatname)

    
    ROI_dict=exi.extract_individuals(Bool_matrix, direc)
    
    linmatrix=np.load('../data/datasets/'+direc+linmatname)
    
    newdic=filter_good_ROI_dic(ROI_dict,min_number)
    
    
    check_division(newdic,linmatrix,masks_list,main_dict,comp_threshold)
    
    ROI_index(newdic,masks_list,main_dict)
    
    indexlist=detect_bad_div(newdic,linmatrix,masks_list,thres,thresmin)

    #plot_lineage_tree(newdic,masks_list,main_dict,colormask,direc)
    plot_image_lineage_tree(newdic,masks_list,main_dict,colormask,indexlist,direc)
    np.savez_compressed('../data/datasets/'+direc+ROIdict,newdic,allow_pickle=True)
    np.savez_compressed('../data/datasets/'+direc+indexlistname,indexlist,allow_pickle=True)
    
    
    # os.remove(direc+linmatname)
    # os.remove(direc+boolmatname)
    # os.remove(direc+linkmatname)
    
    

if __name__ == "__main__":
    
    
   
    run_whole_lineage_tree(Directory)


    # for direct in data_set:
    #     print(direct)
        
    #     run_whole_lineage_tree(direct)
        
#%%   
    # manually_regluing(Directory,ROI_dictionary,index_list_name,'1/100','5/',division=False)
    
    
    # dic_name='Main_dictionnary.npz'

    # list_name='masks_list.npz'

    # ROI_dict='ROI_dict.npz'

    # linmat_name='non_trig_Link_matrix.npy'

    # boolmatname="Bool_matrix.npy"

    # linkmatname='Link_matrix.npy'

    # index_list_name='masks_ROI_list.npz'
    
    
    # colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]

    # index_list=np.load(Directory+index_list_name, allow_pickle=True)['arr_0']
    # List_of_masks=np.load(Directory+list_name, allow_pickle=True)['arr_0']
    # main_dict=np.load(Directory+dic_name, allow_pickle=True)['arr_0'].item()
    # newdic=np.load(Directory+ROI_dict, allow_pickle=True)['arr_0'].item()
    # for ROI in newdic.keys():
    #     print(newdic[ROI])
    
    # print(newdic.keys())
    # plot_image_one_ROI('ROI 8',newdic,List_of_masks,main_dict)
    # plot_lineage_tree(newdic,List_of_masks,main_dict,colormask,Directory)
    # plot_image_lineage_tree(newdic,List_of_masks,main_dict,colormask,index_list,Directory)
    



    