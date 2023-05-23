#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:43:22 2023

@author: c.soubrier
"""

import processing as pcs
import numpy as np
from numba import jit
from copy import deepcopy
import tqdm

Directory=  'delta_lamA_03-08-2018/2/'#"delta_parB/18-01-2015/"#"dataset/" #the directory you chose to work on
# different type of datassets with their quality

data_set=[["dataset/",True],["delta_3187/21-02-2019/",True],["delta_3187/19-02-2019/",True],["delta_parB/03-02-2015/",False],["delta_parB/15-11-2014/",False],["delta_parB/18-01-2015/",False],["delta_parB/18-11-2014/",False],["delta_lamA_03-08-2018/1/",True],["delta_lamA_03-08-2018/2/",True],["WT_mc2_55/06-10-2015/",False],["WT_mc2_55/05-10-2015/",False],["WT_mc2_55/30-03-2015/",True],["WT_mc2_55/05-02-2014/",False],["WT_11-02-15/",False,False],["delta_ripA/14-10-2016/",False],["delta_ripA/160330_rip_A_no_inducer/",True],["delta_ripA/160407_ripA_stiffness_septum/",True],["delta_LTD6_04-06-2017/",False]  ]

result_path='Height/Dic_dir/'

dic_name='Main_dictionnary.npy'

list_name='masks_list.npy'

depth_search=5 # max time distance between two comparable frames

#searching distance (pixels) for the optimization algorithm to construct transtion vectors between pictures. 
search_diameter =100 

#fraction of the preserved area to consider child and parent relation for masks (should be less than 0.5 to take into account the division)
surface_thresh=0.34
#fraction of the preserved area to consider child and parent relation for masks (fusioning of 2 masks after division)
final_thresh=0.75




def trans_vector_matrix(dic,max_diff_time):
    fichier=list(dic.keys())[-1]
    maxtime=dic[fichier]['time']
    mat_vec=np.zeros((maxtime+1,2*max_diff_time+1,2),dtype=np.int32)
    mat_ang=np.zeros((maxtime+1,2*max_diff_time+1))
    dic_list=list(dic.keys())
    for number in tqdm.trange(len(dic_list)):
        fichier1=dic_list[number]
        if number<len(dic_list)-1 and not dic[fichier]['masks_error']:
            time1=dic[fichier1]['time']
            masks1=pcs.main_mask(dic[fichier1]['masks'])
            angle1=dic[fichier1]['angle']
            main_centr1=dic[fichier1]['main_centroid']
            i=1
            timer=dic[dic_list[number+i]]['time']
            while abs(timer-time1)<=max_diff_time:
                fichier2=dic_list[number+i]
                if not dic[fichier2]['masks_error']:
                    masks2=pcs.main_mask(dic[fichier2]['masks'])
                    angle2=dic[fichier2]['angle']
                    main_centr2=dic[fichier2]['main_centroid']
                    update_trans_vector_matrix(mat_vec,mat_ang,masks1,angle1,main_centr1,time1,masks2,angle2,main_centr2,timer,max_diff_time)
                i+=1
                if number+i>=len(dic_list):
                    break
                else:
                    timer=dic[dic_list[number+i]]['time']
            
    return mat_vec, mat_ang

#@jit()
def update_trans_vector_matrix(mat_vec,mat_ang,masks1,angle1,main_centr1,time1,masks2,angle2,main_centr2,timer,max_diff_time):      #enlever le rayon et le vecguess
    angle=angle2-angle1
    if angle==0:
        #vecguess=main_centr1-main_centr2
        res=pcs.opt_trans_vec2(masks1,masks2)#,rad,vecguess
        mat_vec[time1,timer-time1+max_diff_time]=res
        mat_vec[timer,time1-timer+max_diff_time]=-res
    else:
        dim1,dim2=np.shape(masks1)
        centerpoint=np.array([dim1//2,dim2//2],dtype=np.int32)
        #vecguess=(pcs.rotation_vector(angle,main_centr1,centerpoint)-main_centr2).astype(np.int32)
        masks1=pcs.rotation_img(angle,masks1,centerpoint)
        res=pcs.opt_trans_vec2(masks1,masks2).astype(np.int32)#,rad,vecguess
        mat_vec[time1,timer-time1+max_diff_time]=res
        mat_ang[time1,timer-time1+max_diff_time]=angle
        mat_ang[timer,time1-timer+max_diff_time]=-angle
        mat_vec[timer,time1-timer+max_diff_time]=(pcs.rotation_vector(-angle,-res,np.array([0,0],dtype=np.int32))).astype(np.int32)



def lineage_matrix(dic,maskslist,mat_vec,mat_ang,max_diff_time,threshold):
    

    mat_dim=len(maskslist)
    mat=np.zeros((mat_dim,mat_dim))
    fichier= list(dic.keys())[0]
    while dic[fichier]['child']!='':
        print(fichier)
        base_time=dic[fichier]['time']
        child_fichier=dic[fichier]['child']
        timer=dic[child_fichier]['time']
        while abs(timer-base_time)<=max_diff_time:
            transfert=mat_vec[base_time,timer-base_time+max_diff_time]
            angle=mat_ang[base_time,timer-base_time+max_diff_time]
            mask_c=dic[child_fichier]['masks']
            area_c=dic[child_fichier]['area']
            mask_p=dic[fichier]['masks']
            area_p=dic[fichier]['area']
            
            mask_c=pcs.mask_transfert(mask_c,transfert)
            
            if angle!=0:            #check on real data
                dim1,dim2=np.shape(mask_p)
                centerpoint=np.array([dim1//2,dim2//2],dtype=np.int32)
                mask_p=pcs.rotation_img(angle,mask_p,centerpoint)
            
            links_p,links_c =comparision_mask_score(mask_c,mask_p,area_c,area_p,threshold)
            len_p,len_c=np.shape(links_p)
            # for the matrix the entry [i,j] means the intersection of mask i and j divided by area of mask i
            
            for k in range(len_p):
                for l in range(len_c):
                    i=dic[fichier]["mask_list"][k]
                    j=dic[child_fichier]["mask_list"][l]
                    mat[i,j]=links_p[k,l]
                    j=dic[fichier]["mask_list"][k]
                    i=dic[child_fichier]["mask_list"][l]
                    mat[i,j]=links_c[l,k]
                    
            child_fichier=dic[child_fichier]['child']
            if  child_fichier=='':
                break
            timer=dic[child_fichier]['time']
            
        fichier=dic[fichier]['child']
    return mat

@jit 
def comparision_mask_score(mask_c,mask_p,area_c,area_p,threshold):
    number_mask_c=len(area_c)
    number_mask_p=len(area_p)
    dim1,dim2=np.shape(mask_p)
    result_p=np.zeros((number_mask_p,number_mask_c))
    result_c=np.zeros((number_mask_c,number_mask_p))
    for j in range(1,number_mask_c+1):
        for i in range(1,number_mask_p+1):
            area=0
            for k in range(dim1):
                for l in range(dim2):
                    if mask_c[k,l]==j and mask_p[k,l]==i:
                        area+=1
            if area/area_c[j-1]>=threshold:
                result_c[j-1,i-1]=area/area_c[j-1]
            if area/area_p[i-1]>=threshold:
                result_p[i-1,j-1]=area/area_p[i-1]
    return result_p,result_c

def clean_matrix(lin_mat,dic,maskslist,max_diff_time,thres):
    mat_dim=len(maskslist)
    newmat=np.zeros((mat_dim,mat_dim),dtype=np.int8)
    for i in range(mat_dim):
        base_time=dic[maskslist[i][2]]['time']
        time_list=[]
        for k in range(max_diff_time):
            time_list.append([])
        if i<mat_dim:
            for j in range(i+1,mat_dim):
                if lin_mat[i,j]>thres/2:      #no to put 0, float precision problem can appear
                    time_list[dic[maskslist[j][2]]['time']-base_time-1].append(j)
            
            for index in range(len(time_list)):
                element=time_list[index]
                if len(element)==1:
                    if lin_mat[i,element[0]]>thres and lin_mat[element[0],i]>thres:
                        newmat[i,element[0]]= index+1
                elif len(element)==2:
                    if lin_mat[i,element[0]]+lin_mat[i,element[1]]>thres and lin_mat[element[0],i]>thres and lin_mat[element[1],i]>thres:
                        newmat[i,element[0]]= index+1
                        newmat[i,element[1]]= index+1

    finalmatrix=Bool_from_linkmatrix(newmat,mat_dim,max_diff_time) #
    return finalmatrix,newmat


'''         First naive version of the boolean matrix
@jit
def updateclean_matrix(newmat,mat_dim):
    tot_max=np.max(newmat)+1
    finalmat=np.zeros((mat_dim,mat_dim),dtype=np.bool_)
    for i in range(mat_dim):
        for j in range(mat_dim):
            if newmat[i,j]==0:
                newmat[i,j]=tot_max
    for i in range(mat_dim):
        min_col=np.min(newmat[i,:])
        for j in range (i+1,mat_dim):
            finalmat[i,j]=(newmat[i,j]==min_col and min_col<tot_max)        
    for i in range(mat_dim):
        for j in range(mat_dim):
            if newmat[i,j]==tot_max:
                newmat[i,j]=0
    return finalmat
'''   


'''
Deriving the boolean matrix from the link matrix. The bollean matrix maximizes the size of trees


'''

def Bool_from_linkmatrix(linkmat,dim,max_diff_time):
    newmat=np.zeros((dim,dim),dtype=bool)
    forwardlinks,backwardlinks,rootslist=detekt_roots(linkmat,dim) #detekts roots and transform the matrix into an adjacence list
    endpoints=detekt_end_leafs(forwardlinks,backwardlinks,linkmat,dim,max_diff_time)
    #print(endpoints)
    final_links=longuest_path(forwardlinks,backwardlinks,rootslist,dim)
    for point in endpoints:
        update_bool_mat(final_links[point],newmat)
    return newmat


def update_bool_mat(link_list,mat):
    len_lis=len(link_list)
    if len_lis>=2:
        links=deepcopy(link_list)
        links.reverse()
        for num in range(1,len_lis):
            i=links[num]
            j=links[num-1]
            if mat[i,j]:
                break
            else:
                mat[i,j]=True
        
    

def detekt_roots(linkmat,dim):
    
    forwardlinks=[]
    backwardlinks=[]
    for i in range(dim):
        forwardlinks.append([])
        backwardlinks.append([])
    rootslist=[]
    
    
    for i in range(dim):
        if np.max(linkmat[i,:])>0:
            for j in range(i+1,dim):
                if linkmat[i,j]>0:
                    forwardlinks[i].append(j)
                    backwardlinks[j].append(i)
            if np.max(linkmat[:,i])==0:
                rootslist.append(i)
        elif np.max(linkmat[:,i])==0:
            rootslist.append(i)
            
            
    return forwardlinks,backwardlinks,rootslist

def detekt_end_leafs(forwardlinks,backwardlinks,linkmat,dim,depth):
    res=[]
    for i in range(dim):
        if forwardlinks[i]==[] and backwardlinks[i]!=[]:
            #print(i)
            ancestors=list_ancestors(i,backwardlinks,linkmat,depth)
            #print(ancestors)
            children=list_children(ancestors,forwardlinks,linkmat,depth+1)
            #print(children)
            if len(ancestors) > 2 and not max(children)>i:          #having more than 2 ancestors means that a real division has been detected
                res.append(i)

    return res

def end_leafs(linkmat,dim):
    leafslist=[]
    for i in range(dim):
        if np.max(linkmat[i,:])==0:
            leafslist.append(i)
    return leafslist


def list_ancestors(i,backwardlinks,linkmat,depth):
    res=[i]
    if depth<1:
        return res
    else:
        for indiv in backwardlinks[i]:
            subdepth=linkmat[indiv ,i]
            if subdepth>=1 and subdepth<=depth and np.count_nonzero(linkmat[indiv,:]==subdepth)==1:
                res+=list_ancestors(indiv, backwardlinks,linkmat,depth-subdepth)
        return list(set(res))

def list_children(ancestors,forwardlinks,linkmat,depth):
    res=deepcopy(ancestors)
    if depth<1:
        return res
    else:
        for indiv in ancestors:
            for link in forwardlinks[indiv]:
                subdepth=linkmat[indiv,link]
                if subdepth>=1 and subdepth<=depth:
                    res+=list_children([link],forwardlinks,linkmat,depth-subdepth)
            
        return list(set(res))
            

def longuest_path(forwardlinks,backwardlinks,rootslist,dim):
    path=[]
    value=np.zeros(dim,dtype=np.int32)
    for i in range(dim):
        path.append([])
    for root in rootslist:
        path[root]=[root]
        value[root]=1
        
        
    for iteration in range(dim):
        value_iter,path_iter=update_longest_path(iteration,backwardlinks,value,path)
        value[iteration]=value_iter
        path[iteration]=path_iter
        
    return path
        
    
def update_longest_path(iteration,backwardlinks,value,path):
    if value[iteration]>0:
        return value[iteration],path[iteration]
    else:
        parents=backwardlinks[iteration]
        parent_number=len(parents)
        count=0
        finalpath=[]
        for i in range(parent_number):
            parent_value,parent_path=update_longest_path(parents[i],backwardlinks,value,path)
            if parent_value>=count:
                count,finalpath=parent_value,parent_path
        value[iteration]=count+1
        path[iteration]=finalpath+[iteration]
        return value[iteration],path[iteration]
        
    


def Final_lineage_tree(direc,resultpath,dicname,listname,max_diff_time,surfthresh,finthres):
    masks_list=np.load(direc+resultpath+listname, allow_pickle=True)
    main_dict=np.load(direc+resultpath+dicname, allow_pickle=True).item()
    vector_matrix, angle_matrix=trans_vector_matrix(main_dict,max_diff_time) #translation vector and rotation angle between the different frames
    print(1)
    lin_mat=lineage_matrix(main_dict,masks_list,vector_matrix, angle_matrix,max_diff_time,surfthresh)
    print(2)
    Bool_mat,Link_mat=clean_matrix(lin_mat,main_dict,masks_list,max_diff_time,finthres)
    print(3)
    np.save(direc+resultpath+'Bool_matrix',Bool_mat)
    np.save(direc+resultpath+'Link_matrix',Link_mat)
    np.save(direc+resultpath+'non_trig_Link_matrix',lin_mat)
    

if __name__ == "__main__":
    Final_lineage_tree(Directory,result_path,dic_name,list_name,depth_search,surface_thresh,final_thresh)
    # for direc in data_set:
    #     print(direc[0])
    #     Final_lineage_tree(direc[0],result_path,dic_name,list_name,depth_search,search_diameter,surface_thresh,final_thresh)
    # A=np.load(Directory+result_path+'Bool_matrix.npy',).astype(int)
    # B=np.load(Directory+result_path+'Link_matrix.npy')
    # C=np.load(Directory+result_path+'ROI_dict.npy',allow_pickle=True).item()
    # D=np.load(Directory+result_path+'non_trig_Link_matrix.npy')
    # masks_list=np.load(Directory+result_path+list_name, allow_pickle=True)
    # main_dict=np.load(Directory+result_path+dic_name, allow_pickle=True).item()
    # Bool_mat,Link_mat,final_links=clean_matrix(D,main_dict,masks_list,depth_search*3,final_thresh)
    # np.save(Directory+result_path+'Bool_matrix.npy',Bool_mat)


