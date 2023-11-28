#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:15:33 2023

@author: c.soubrier
"""


import sys
import os

from sklearn import manifold
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from scaled_parameters import get_scaled_parameters

from DeCOr_MDS import DeCOr_MDS
from centerline_analysis_v2 import dist_centerline, scaling_centerlines,L2_score,comparison_centerline





# def plot_centerline_align(center_list, invers_mat,delta_mat,list0,list1,dicname,diclistname,datadirec):
#     initdata0=''
#     initdata1=''
#     for elem0 in list0:
#         for elem1 in list1:
#             data0=center_list[elem0][0]
#             data1=center_list[elem1][0]
#             if data0!=initdata0:
#                 dic0=np.load(os.path.join(datadirec, data0, dicname), allow_pickle=True)['arr_0'].item()
#                 mask_list0=np.load(os.path.join(datadirec, data0, diclistname), allow_pickle=True)['arr_0']
#                 initdata0=data0
#             if data1!=initdata1:
#                 dic1=np.load(os.path.join(datadirec, data1, dicname), allow_pickle=True)['arr_0'].item()
#                 mask_list1=np.load(os.path.join(datadirec, data1, diclistname), allow_pickle=True)['arr_0']
#                 initdata1=data1
#             plot_2_center(dic0,mask_list0,elem0,dic1,mask_list1,elem1,center_list,invers_mat,delta_mat)
        
        

# def plot_2_center(dic0,mask_list0,num0,dic1,mask_list1,num1,center_list,invers_mat,delta_mat):

#     fichier0,maskind0=mask_list0[int(center_list[num0][1])][2:]
#     centerline0=dic0[fichier0]['centerlines'][maskind0-1]
#     size0=dic0[fichier0]['resolution']
#     img0=np.load(dic0[fichier0]['adress'])['Height_fwd']
#     line_data0=dist_centerline(centerline0,img0)
    
    
#     fichier1,maskind1=mask_list1[int(center_list[num1][1])][2:]
#     centerline1=dic1[fichier1]['centerlines'][maskind1-1]
#     size1=dic1[fichier1]['resolution']
#     img1=np.load(dic1[fichier1]['adress'])['Height_fwd']
#     line_data1=dist_centerline(centerline1,img1)
    
   
#     if line_data0[2] or line_data1[2]:
#         print('problem centerline')
#     else:
#         height0,dist0=line_data0[:2]
#         height1,dist1=line_data1[:2]
        
#         print(comparison_centerline(height0, height1, dist0, dist1, size0, size1, 0.5, 0.1, 0.1,None))
        
#         (phy_height0,phy_height1,pix_len0,pix_len1)=scaling_centerlines(height0,height1,dist0,dist1,size0,size1)
#         size=min(size0,size1)
#         if invers_mat[num0,num1]:
#             phy_height1=phy_height1[::-1]
#             print('inverted')
            
#         if pix_len0>pix_len1:
#             print('good')
#             plt.figure()
#             avg0=np.average(phy_height0[len(phy_height0)//7:6*len(phy_height0)//7])
#             plt.plot(np.linspace(0,(pix_len0-1)*size,pix_len0),phy_height0,color='r')

#             avg1=np.average(phy_height1[len(phy_height1)//7:6*len(phy_height1)//7])
#             plt.plot(np.linspace((delta_mat[num0,num1]-pix_len1//10)*size,(delta_mat[num0,num1]-pix_len1//10+pix_len1-1)*size,pix_len1),phy_height1+np.ones(len(phy_height1))*(avg0-avg1),color='k')
#         else:
#             plt.figure()
#             avg0=np.average(phy_height0[len(phy_height0)//7:6*len(phy_height0)//7])
#             avg1=np.average(phy_height1[len(phy_height1)//7:6*len(phy_height1)//7])
#             plt.plot(np.linspace(-pix_len0//10*size,(pix_len0-1-pix_len0//10)*size,pix_len0),phy_height0,color='r')

#             plt.plot(np.linspace((delta_mat[num0,num1])*size,(delta_mat[num0,num1]+pix_len1-1)*size,pix_len1),phy_height1+np.ones(len(phy_height1))*(avg0-avg1),color='k')
#             plt.xlabel(r'length $(\mu m)$')
#             plt.ylabel(r'height $(nm)$')



def run_mds_plot(center_list,dist_mat,dicname,listindnam,title,datadirec, mask=None, ROI_list=None):
    
    params=get_scaled_parameters(plot=True)
    colormask=params["masks_colors"] 
    if mask is not None:
        print (len(dist_mat), len (mask))
        dist_mat=dist_mat[mask,:][:,mask]
        center_list=center_list[mask]
    
    print(len(center_list)) 
    res=DeCOr_MDS(dist_mat,4,2,4)
    outliers=res[0]
    print('dimension : ', res[1])
    mask=np.ones(len(dist_mat),dtype=bool)
    for elem in outliers:
        mask[elem]=False
        
    
    dist_mat_prime=dist_mat[mask,:][:,mask]
    center_list_prime=center_list[mask]
    
    
    mds = manifold.MDS(n_components=res[1], random_state = 1, dissimilarity="precomputed")
    pos = mds.fit(dist_mat_prime).embedding_

    data_list=np.array(list(set([elem[1] for elem in  center_list_prime])))
    print(len(pos),len(center_list_prime),data_list)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(f"MDS of {title} with lineage")

    oldname=''
    for i in trange(len(pos)):
        infos=center_list_prime[i]
        change_data=False
        if i==0 or oldname!=infos[0]:
            change_data=True
            oldname=infos[0]
            dic=np.load(os.path.join(datadirec, infos[0], dicname), allow_pickle=True)['arr_0'].item()
            # ROI_dict=np.load(datadirec+infos[0]+ROI_name, allow_pickle=True)['arr_0'].item()
            mask_list=np.load(os.path.join(datadirec, infos[0], 'masks_list.npz'), allow_pickle=True)['arr_0']
            ROI_mask_list=np.load(os.path.join(datadirec, infos[0], listindnam), allow_pickle=True)['arr_0']
            
        # col=int(ROI_mask_list[int(infos[1])][0])
        # c=np.array(colormask[col%len(colormask)])/255
        
        col_data=int(np.argmax(data_list==infos[0]))
        c=np.array([colormask[col_data%len(colormask)]])/255
        
        frame=mask_list[int(infos[1])][2]
        time=dic[frame]['time']
        
        
        if ROI_list is None:
            # collin='red'
            # if time>700:
            #     collin='blue'
            if not change_data:
                if ROI_mask_list[int(infos[1])][2]==ROI_mask_list[int(center_list_prime[i-1][1])][2]:
                    # ax.plot3D([pos[i-1, 0],pos[i, 0]], [pos[i-1, 1],pos[i, 1]],[pos[i-1, 2],pos[i, 2]],c=c,linewidth=0.5)#c=collin
                    ax.scatter(pos[i, 0], pos[i, 1],pos[i, 2],c=c,s=2)
            else :
                ax.scatter(pos[i, 0], pos[i, 1],pos[i, 2],c=c,s=2)#collin
        
        
        
        else :
            if ROI_mask_list[int(infos[1])][2] in ROI_list:
                if not change_data:
                    if ROI_mask_list[int(infos[1])][2]==ROI_mask_list[int(center_list_prime[i-1][1])][2]:
                        ax.plot3D([pos[i-1, 0],pos[i, 0]], [pos[i-1, 1],pos[i, 1]],[pos[i-1, 2],pos[i, 2]],c=c,linewidth=0.5)#c=collin
                        # ax.scatter(pos[i, 0], pos[i, 1],pos[i, 2],c=c)
                else :
                    ax.scatter(pos[i, 0], pos[i, 1],pos[i, 2],c=c,s=2)#collin
        


    # ROI_outlier=set([ROI_mask_list[i][2] for i in outliers])
    # print('Outliers : ',ROI_outlier )


def run_mds_plot_2_data_time(center_list,dist_mat,base_data,evoldata,dicname,masklistname,datadirec):
    lenlis=len(center_list)
    mask=np.zeros(lenlis,dtype=bool)
    for i in range(lenlis):
        if center_list[i,0] in [base_data,evoldata]:
                mask[i]=1
    
    print (len(dist_mat), len (mask))
    dist_mat=dist_mat[mask,:][:,mask]
        
        
    center_list=center_list[mask]
    print(len(center_list)) 
    res=DeCOr_MDS(dist_mat,4,2,4)
    outliers=res[0]
    print('dimension : ', res[1])
    mask=np.ones(len(dist_mat),dtype=bool)
    for elem in outliers:
        mask[elem]=False
        
    
    dist_mat_prime=dist_mat[mask,:][:,mask]
    center_list_prime=center_list[mask]
    
    list_WT=np.zeros(len(center_list_prime),dtype=bool)
    for i in range(len(center_list_prime)):
        if center_list_prime[i,0]==base_data:
            list_WT[i]=1
    
    
    mds = manifold.MDS(n_components=res[1], random_state = 1, dissimilarity="precomputed")
    pos = mds.fit(dist_mat_prime).embedding_

    data_list=np.array(list(set([elem[0] for elem in  center_list_prime])))
    print(len(pos),len(center_list_prime),data_list)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(f"MDS of {base_data[:10]} vs {evoldata[:10]}")


    c=[0.7,0.7,0.7]
    pos_WT=pos[list_WT]
    ax.scatter(pos_WT[:, 0], pos_WT[:, 1],pos_WT[:, 2],color=c,s=2,label=base_data)
    
    pos_mu=pos[np.logical_not(list_WT)]
    new_center_list=center_list_prime[np.logical_not(list_WT)]
    timing=np.zeros(len(new_center_list))
    dic=np.load(os.path.join(datadirec, evoldata, dicname), allow_pickle=True)['arr_0'].item()
    mask_list=np.load(os.path.join(datadirec, evoldata, masklistname), allow_pickle=True)['arr_0']
    
    for i in trange(len(new_center_list)):
        frame=mask_list[int(new_center_list[i,1])][2]
        timing[i]=dic[frame]['time']
    
    p=ax.scatter(pos_mu[:, 0],pos_mu[:, 1],pos_mu[:, 2],c=timing,s=2,cmap='plasma')
    cbar=plt.colorbar(p)
    cbar.set_label('time (mn), '+evoldata[:6])
    
    ax.xaxis.set_ticklabels([''])
    ax.yaxis.set_ticklabels([''])
    ax.zaxis.set_ticklabels([''])
    plt.legend()

    
    
def run_mds_plot_2_data_cluster(center_list,dist_mat,base_data,evoldata):
    lenlis=len(center_list)
    mask=np.zeros(lenlis,dtype=bool)
    for i in range(lenlis):
        if center_list[i,0] in [base_data,evoldata]:
                mask[i]=1
    
    print (len(dist_mat), len (mask))
    dist_mat=dist_mat[mask,:][:,mask]
        
        
    center_list=center_list[mask]
    print(len(center_list)) 
    res=DeCOr_MDS(dist_mat,4,2,4)
    outliers=res[0]
    print('dimension : ', res[1])
    mask=np.ones(len(dist_mat),dtype=bool)
    for elem in outliers:
        mask[elem]=False
        
    
    dist_mat_prime=dist_mat[mask,:][:,mask]
    center_list_prime=center_list[mask]
    
    list_WT=np.zeros(len(center_list_prime),dtype=bool)
    for i in range(len(center_list_prime)):
        if center_list_prime[i,0]==base_data:
            list_WT[i]=1
    
    
    mds = manifold.MDS(n_components=res[1], random_state = 1, dissimilarity="precomputed")
    pos = mds.fit(dist_mat_prime).embedding_

    data_list=np.array(list(set([elem[0] for elem in  center_list_prime])))
    print(len(pos),len(center_list_prime),data_list)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title( f"MDS of {base_data[:10]} vs {evoldata[:10]}" )


    c=[0,0.45,0.74]
    pos_WT=pos[list_WT]
    ax.scatter(pos_WT[:, 0], pos_WT[:, 1],pos_WT[:, 2],color=c,s=2,label=base_data)
    
    c=[0.7,0,0]
    pos_mu=pos[np.logical_not(list_WT)]
    ax.scatter(pos_mu[:, 0], pos_mu[:, 1],pos_mu[:, 2],color=c,s=2,label=evoldata)
    
    plt.legend()

    
    kmeans = cluster.KMeans(n_clusters=2, random_state=0, n_init=100).fit(pos)
    ind_clus1=kmeans.labels_==0
    number_k1=sum(ind_clus1)
    number_k1_WT=sum(np.logical_and(ind_clus1,list_WT))
    number_k1_mu=sum(np.logical_and(ind_clus1,np.logical_not(list_WT)))
    
    
    ind_clus2=kmeans.labels_==1
    number_k2=sum(ind_clus2)
    number_k2_WT=sum(np.logical_and(ind_clus2,list_WT))
    number_k2_mu=sum(np.logical_and(ind_clus2,np.logical_not(list_WT)))
    
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Clustering of the MDS")
    
    c=[0,0.45,0.74]
    pos_k1=pos[ind_clus1]
    ax.scatter(pos_k1[:, 0], pos_k1[:, 1],pos_k1[:, 2],color=c,s=2,label=f'k1 : {number_k1_WT/number_k1} {base_data[:3]}, {number_k1_mu/number_k1} {evoldata[:3]}')
    
    c=[0.7,0,0]
    pos_k2=pos[ind_clus2]
    ax.scatter(pos_k2[:, 0], pos_k2[:, 1],pos_k2[:, 2],color=c,s=2,label=f'k2 : {number_k2_WT/number_k2} {base_data[:3]}, {number_k2_mu/number_k2} {evoldata[:3]}')
    plt.legend()

    
    
    
    
    
    
    
    
    
    
    
    
def run_mds_plot_INH_time(center_list,dist_mat,dicname,datadirec):
    
    
    dic=np.load(os.path.join(datadirec, 'WT_INH_700min_2014/', dicname), allow_pickle=True)['arr_0'].item()
    # ROI_dict=np.load(infos[1]+ROI_name, allow_pickle=True)['arr_0'].item()
    mask_list=np.load(os.path.join(datadirec, 'WT_INH_700min_2014/', 'masks_list.npz'), allow_pickle=True)['arr_0']
    
    
    
    
    lenlis=len(center_list)
    mask=np.zeros(lenlis,dtype=bool)
    for i in range(lenlis):
        if center_list[i,0] =='WT_INH_700min_2014':
            frame=mask_list[int(center_list[i,1])][2]
            time=dic[frame]['time']
            if not 1400<time:
                mask[i]=1
    
    print (len(dist_mat), len (mask))
    dist_mat=dist_mat[mask,:][:,mask]
        
        
    center_list=center_list[mask]
    print(len(center_list)) 
    res=DeCOr_MDS(dist_mat,4,2,4)
    outliers=res[0]
    print('dimension : ', res[1])
    mask=np.ones(len(dist_mat),dtype=bool)
    for elem in outliers:
        mask[elem]=False
        
    
    dist_mat_prime=dist_mat[mask,:][:,mask]
    center_list_prime=center_list[mask]
    
    
    
    list_WT=np.zeros(len(center_list_prime),dtype=bool)
    for i in range(len(center_list_prime)):
        frame=mask_list[int(center_list_prime[i,1])][2]
        time=dic[frame]['time']
        if time<=700:
            list_WT[i]=1
    
    
    
    mds = manifold.MDS(n_components=res[1], random_state = 1, dissimilarity="precomputed")
    pos = mds.fit(dist_mat_prime).embedding_

    data_list=np.array(list(set([elem[0] for elem in  center_list_prime])))
    print(len(pos),len(center_list_prime),data_list)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("MDS before/after INH ")


    c=[0.7,0.7,0.7]
    pos_WT=pos[list_WT]
    ax.scatter(pos_WT[:, 0], pos_WT[:, 1],pos_WT[:, 2],color=c,s=2,label='before INH')
    
    centro_WT=np.array([np.mean(pos_WT[:, 0]),np.mean(pos_WT[:, 1]),np.mean(pos_WT[:, 2])])
    dist_WT=np.zeros(len(pos_WT))
    for i in range(len(pos_WT)):
        dist_WT[i]=np.linalg.norm(pos_WT[i, :]-centro_WT)
    
    
    pos_mu=pos[np.logical_not(list_WT)]
    new_center_list=center_list_prime[np.logical_not(list_WT)]
    timing=np.zeros(len(new_center_list))
    
    centro_mu=np.array([np.mean(pos_mu[:, 0]),np.mean(pos_mu[:, 1]),np.mean(pos_mu[:, 2])])
    dist_mu=np.zeros(len(pos_mu))
    for i in range(len(pos_mu)):
        dist_mu[i]=np.linalg.norm(pos_mu[i, :]-centro_mu)
    
    for i in trange(len(new_center_list)):
        frame=mask_list[int(new_center_list[i,1])][2]
        timing[i]=dic[frame]['time']
    
    p=ax.scatter(pos_mu[:, 0],pos_mu[:, 1],pos_mu[:, 2],c=timing,s=2,cmap='plasma')
    cbar=plt.colorbar(p)
    cbar.set_label('time (mn), after INH')
    plt.legend()
    ax.xaxis.set_ticklabels([''])
    ax.yaxis.set_ticklabels([''])
    ax.zaxis.set_ticklabels([''])

    
    plt.figure()
    res=dist_WT/np.sum(dist_WT)
    mean=np.average(res)
    std=np.std(res)
    plt.hist(res, 100, alpha=0.3, color="blue", label='before INH : '+"mean "+str(round(mean))+', std '+str(round(std)))
    
    plt.axvline(mean, color="blue", label='mean before INH')
    
    res=dist_mu/np.sum(dist_mu)
    mean=np.average(res)
    std=np.std(res)
    plt.hist(res, 100, alpha=0.3, color="red", label='after INH : '+"mean "+str(round(mean))+', std'+str(round(std)))
    plt.axvline(mean, color="red", label='mean after INH')
    plt.legend(loc='upper right')
    plt.xlabel('distance in the mds')
    plt.ylabel('normalized distribution')
    plt.title('distance distribution')



def plot_growth_rate_INH(dicname,masklistname,Roiname,datadirect):
    data='WT_INH_700min_2014/'
    dic=np.load(os.path.join(datadirect, data, dicname), allow_pickle=True)['arr_0'].item()
    mask_list=np.load(os.path.join(datadirect, data, masklistname), allow_pickle=True)['arr_0']
    ROI_dic=np.load(os.path.join(datadirect, data, Roiname), allow_pickle=True)['arr_0'].item()
    
    size=dic[list(dic.keys())[0]]['resolution']
    
    gr_before=[]
    gr_indef=[]
    gr_after=[]
    for ROI in ROI_dic.keys():
        if  len(ROI_dic[ROI]["Mask IDs"])>7:
            first_elem=ROI_dic[ROI]["Mask IDs"][0]
            last_elem=ROI_dic[ROI]["Mask IDs"][-1]
            
            
            fichier0,maskind0=mask_list[first_elem][2:]
            time0=dic[fichier0]['time']
            centerline0=dic[fichier0]['centerlines'][maskind0-1]
            img0=np.load(dic[fichier0]['adress'])['Height_fwd']
            
            len0=dist_centerline(centerline0,img0)[1][-1]*size
            
            fichier1,maskind1=mask_list[last_elem][2:]
            time1=dic[fichier1]['time']
            centerline1=dic[fichier1]['centerlines'][maskind1-1]
            img1=np.load(dic[fichier1]['adress'])['Height_fwd']
            
            len1=dist_centerline(centerline1,img1)[1][-1]*size
            
            if len1-len0>=0:
                if time1<700:
                    gr_before.append((len1-len0)/(time1-time0)*1000)
                elif time0>700:
                    gr_after.append((len1-len0)/(time1-time0)*1000)
                else:
                    gr_indef.append((len1-len0)/(time1-time0)*1000)
                
    plt.figure()
    plt.title('Growth rate')
    
    res=np.array(gr_before)
    mean=np.average(res)
    std=np.std(res)
    plt.hist(res, 20,color='red', alpha=0.3, label='before INH : '+"mean "+str(round(mean,2))+', std '+str(round(std,2)))
    plt.axvline(mean, color='red', label='mean before INH')
    
    res=np.array(gr_after)
    mean=np.average(res)
    std=np.std(res)
    plt.hist(res, 40,color='blue', alpha=0.3, label='after INH'+"mean "+str(round(mean,2))+', std '+str(round(std,2)))
    plt.axvline(mean, color='blue', label='mean after INH')
    res=np.array(gr_indef)
    
    # mean=np.average(res)
    # std=np.std(res)
    # plt.hist(res, 30,color='C2', alpha=0.2, label='during INH'+"mean "+str(round(mean,2))+', std '+str(round(std,2)))
    # plt.axvline(mean, color='C2', label='mean during INH')
    plt.xlabel(r'growth rate ($n m / mn$)')
    plt.ylabel('number of cells')
    plt.legend()
    
    
    
def main():
    params = get_scaled_parameters(paths_and_names=True)
    title='WT30-03-2015 vs delta_LTD6' 
    
    data_direc = params["main_data_direc"]

    dir_cent = params["dir_res_centerlines"] 

    A=np.load(os.path.join(dir_cent, params['centerline_list']))
    B=np.load(os.path.join(dir_cent, params['distance_matrix']))
    C=np.load(os.path.join(dir_cent, params['inversion_matrix']))
    D=np.load(os.path.join(dir_cent, params['delta_matrix']))

    
    
    dic_name = params["main_dict_name"]
    ROI_name = params["roi_dict_name"]
    indexlistname = params['roi_masks_list_name']
    mask_list_name = params["masks_list_name"]

    params = get_scaled_parameters(data_set=True)
    data_set= params["all"]

    # plot_centerline_align(A,C,D,[7],[9],dic_name,mask_list_name,data_direc)
   

    run_mds_plot_INH_time(A,B,dic_name,data_direc)
    plot_growth_rate_INH(dic_name,mask_list_name,ROI_name,data_direc)

    lenlis=len(A)
    mask=np.zeros(lenlis,dtype=bool)
    for i in range(lenlis):
        olddata=''
        if A[i,0] in data_set:
            if A[i,0]!=olddata:
                ROI_dic=np.load(os.path.join(data_direc, A[i,0], ROI_name), allow_pickle=True)['arr_0'].item()
            if len(ROI_dic[A[i,2]]['Mask IDs'])>7:
                mask[i]=1

    # run_mds_plot(A,B,dic_name,indexlistname,title,data_direc,mask=mask)

    run_mds_plot_2_data_time(A,B,"WT_mc2_55/30-03-2015",'WT_INH_700min_2014',dic_name,mask_list_name,data_direc)
    run_mds_plot_2_data_cluster(A,B,"WT_mc2_55/30-03-2015","delta_parB/18-01-2015")

    plt.show()
    
    
    
if __name__ == "__main__":
    
    main()


    
    



    
    








