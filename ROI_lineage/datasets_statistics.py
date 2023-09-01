#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:33:10 2023

@author: c.soubrier
"""

import numpy as np
import matplotlib.pyplot as plt
import centerline_analysis_v2 as ca
import tqdm


ROI_name='ROI_dict.npz' 



# data_set=[["dataset/",True],["delta_3187/21-02-2019/",True],["delta_3187/19-02-2019/",True],["delta_parB/03-02-2015/",False],["delta_parB/15-11-2014/",False],["delta_parB/18-01-2015/",False],["delta_parB/18-11-2014/",False],["delta_lamA_03-08-2018/1/",True],["delta_lamA_03-08-2018/2/",True],["WT_mc2_55/06-10-2015/",False],["WT_mc2_55/05-10-2015/",False],["WT_mc2_55/30-03-2015/",True],["WT_mc2_55/05-02-2014/",False],["WT_11-02-15/",False,False],["delta_ripA/14-10-2016/",False],["delta_ripA/160330_rip_A_no_inducer/",True],["delta_ripA/160407_ripA_stiffness_septum/",True],["delta_LTD6_04-06-2017/",False]  ]
# data_set2=['delta_lamA_03-08-2018/','delta_LTD6_04-06-2017/',"delta_parB/03-02-2015/","delta_parB/15-11-2014/","delta_parB/18-01-2015/","delta_parB/18-11-2014/","delta_ripA/14-10-2016/","WT_mc2_55/06-10-2015/","WT_mc2_55/30-03-2015/","WT_mc2_55/03-09-2014/",'WT_INH_700min_2014/','WT_CCCP_irrigation_2016/','WT_filamentation_cipro_2015/']

data_set2="WT_mc2_55/06-10-2015/","WT_mc2_55/30-03-2015/","WT_mc2_55/03-09-2014/"
dic_name='Main_dictionnary.npz'

list_name='masks_list.npz'


def classifying_ROI(direct_list,ROIdicname): #with parent and daughter cells
    print('classifying_ROI')
    wholeROI=[]
    rootsonly=[]
    leafonly=[]
    nonconnected=[]
    numberdivision=0
    for i in tqdm.trange(len(direct_list)):
        direct=direct_list[i]
        ROIdic=np.load(direct+ROIdicname, allow_pickle=True)['arr_0'].item()
        for ROI in ROIdic.keys():
            if ROIdic[ROI]['Parent']=='':
                if ROIdic[ROI]['Children']==[]:
                    nonconnected.append([ROI,direct])
                else:
                    rootsonly.append([ROI,direct])
                    numberdivision+=1
                    
            else:
                if ROIdic[ROI]['Children']==[]:
                    leafonly.append([ROI,direct])
                else:
                    wholeROI.append([ROI,direct])
                    numberdivision+=1
    return wholeROI,rootsonly,leafonly, nonconnected,numberdivision


def stat_centerline_surface(wholeROI,ROIdicname,dicname,masklistname):
    centerlen=[]
    mask_surf=[]
    print('stat_centerline_surface')
    for i in tqdm.trange(len(wholeROI)):
        elem =wholeROI[i]
        ROI,direct=elem
        ROIdic=np.load(direct+ROIdicname, allow_pickle=True)['arr_0'].item()
        maindict=np.load(direct+dicname, allow_pickle=True)['arr_0'].item()
        masklist=np.load(direct+masklistname, allow_pickle=True)['arr_0']
        centdist=[]
        subsurf=[]
        for mask in ROIdic[ROI]['Mask IDs']:
            file=masklist[mask][2]
            index=masklist[mask][3]
            
            img=np.load(maindict[file]['adress'])['Height_fwd']
            centerline=maindict[file]['centerlines'][index-1]
            area=maindict[file]['area'][index-1]
            resolution=maindict[file]['resolution']
            subsurf.append(area*resolution**2)
            # print(img,centerline)
            if centerline.size  :
                (height1,dist1)=ca.dist_centerline(centerline,img)
                totdist=dist1[-1]
                centdist.append(totdist*resolution)
            #voulme=
        centerlen.append(centdist)
        mask_surf.append(subsurf)
    return(centerlen,mask_surf)

def stat_end_ROI(rootsonly,ROIdicname,dicname,masklistname):
    centerlen=[]
    mask_surf=[]
    print('stat_end_ROI')
    for i in tqdm.trange(len(rootsonly)):
        elem =rootsonly[i]
        
        ROI,direct=elem
        ROIdic=np.load(direct+ROIdicname, allow_pickle=True)['arr_0'].item()
        maindict=np.load(direct+dicname, allow_pickle=True)['arr_0'].item()
        masklist=np.load(direct+masklistname, allow_pickle=True)['arr_0']
        
        mask=ROIdic[ROI]['Mask IDs'][-1]
        file=masklist[mask][2]
        index=masklist[mask][3]
        
        img=np.load(maindict[file]['adress'])['Height_fwd']
        centerline=maindict[file]['centerlines'][index-1]
        area=maindict[file]['area'][index-1]
        resolution=maindict[file]['resolution']
        if centerline.size :
            (height1,dist1)=ca.dist_centerline(centerline,img)
            totdist=dist1[-1]
            centerlen.append(totdist*resolution)
        
        mask_surf.append(area*resolution**2)
    return(centerlen,mask_surf)


def stat_begin_ROI(leafonly,ROIdicname,dicname,masklistname):
    centerlen=[]
    mask_surf=[]
    print('stat_begin_ROI')
    for i in tqdm.trange(len(leafonly)):
        elem =leafonly[i]
        
        ROI,direct=elem
        ROIdic=np.load(direct+ROIdicname, allow_pickle=True)['arr_0'].item()
        maindict=np.load(direct+dicname, allow_pickle=True)['arr_0'].item()
        masklist=np.load(direct+masklistname, allow_pickle=True)['arr_0']
        
        mask=ROIdic[ROI]['Mask IDs'][0]
        file=masklist[mask][2]
        index=masklist[mask][3]
        
        img=np.load(maindict[file]['adress'])['Height_fwd']
        centerline=maindict[file]['centerlines'][index-1]
        area=maindict[file]['area'][index-1]
        resolution=maindict[file]['resolution']
        if centerline.size :
            (height1,dist1)=ca.dist_centerline(centerline,img)
            totdist=dist1[-1]
            centerlen.append(totdist*resolution)
        mask_surf.append(area*resolution**2)
    return(centerlen,mask_surf)

def plotstat(data,title,bar_num=40):
    res=np.array(data)
    mean=np.average(res)
    med=np.median(res)
    q1=np.percentile(res, 25)
    q3=np.percentile(res, 75)
    plt.hist(res, bar_num, density=True, color="grey")
    plt.axvline(mean, color="red", label="mean"+str(mean))
    plt.axvline(med, color="blue", label="median"+str(med))
    plt.axvline(q1, color="green", label="quantiles")
    plt.axvline(q3, color="green")
    plt.legend()
    plt.title(title)
    plt.savefig('stats_'+title, format='jpg')
    plt.show()

def totale_stats(direct_list,ROIdicname,dicname,masklistname):
    
    
    wholeROI,rootsonly,leafonly, nonconnected,numberdivision=classifying_ROI(direct_list,ROIdicname)
    
    
    centerlen,mask_surf=stat_centerline_surface(wholeROI,ROIdicname,dicname,masklistname)
    endcenterlen,endmask_surf=stat_end_ROI(rootsonly,ROIdicname,dicname,masklistname)
    begincenterlen,beginmask_surf=stat_begin_ROI(leafonly,ROIdicname,dicname,masklistname)
    
    
    print('complete ROI : ',len(wholeROI),' roots only : ', len(rootsonly),' leaf only : ',len(leafonly),' non connected : ', len(nonconnected))
    print('number_division',numberdivision)
    stat_list=[[[i for elem in centerlen for i in elem],'Centerlines'],[[i for elem in mask_surf for i in elem],'Surfaces'],[[elem[0] for elem in centerlen]+[elem for elem in begincenterlen],'Centerlines after div'],[[elem[0] for elem in mask_surf]+[elem for elem in beginmask_surf],'Surfaces after div'],[[elem[-1] for elem in centerlen]+[elem for elem in endcenterlen],'Centerlines before div'],[[elem[-1] for elem in mask_surf]+[elem for elem in endmask_surf],'Surfaces before div'],[[elem[-1]/elem[0] for elem in centerlen ],'Centerlines len ratio between 2 div'],[[elem[-1]/elem[0] for elem in  mask_surf],'Surface len ratio between 2 div'],[[np.average(np.array(elem)) for elem in centerlen],'ROI AVG Centerlines'],[[np.average(np.array(elem)) for elem in mask_surf],'ROI AVG Surface']]
    
    for elem in stat_list:
        plotstat(elem[0],elem[1])
    
  

    
   
    
    
            
            
        
def stats_time_evol(direct_list,ROIdicname,dicname,masklistname):
    centerratio=[]
    surfratio=[]
    print('stats_time_evol')
    
    for direct in direct_list:
        ROIdic=np.load(direct+ROIdicname, allow_pickle=True)['arr_0'].item()
        maindict=np.load(direct+dicname, allow_pickle=True)['arr_0'].item()
        masklist=np.load(direct+masklistname, allow_pickle=True)['arr_0']
        for ROI in ROIdic.keys():
            if len(ROIdic[ROI]['Mask IDs'])>5:
                oldfile=masklist[ROIdic[ROI]['Mask IDs'][0]][2]
                
                oldindex=masklist[ROIdic[ROI]['Mask IDs'][0]][3]
                oldtime=maindict[oldfile]['time']
                oldimg=np.load(maindict[oldfile]['adress'])['Height_fwd']
                oldcenterline=maindict[oldfile]['centerlines'][oldindex-1]
                
                newfile=masklist[ROIdic[ROI]['Mask IDs'][-1]][2]
                
                newindex=masklist[ROIdic[ROI]['Mask IDs'][-1]][3]
                newtime=maindict[newfile]['time']
                newimg=np.load(maindict[newfile]['adress'])['Height_fwd']
                newcenterline=maindict[newfile]['centerlines'][newindex-1]
                
                if newcenterline.size and oldcenterline.size:
                    (height1,dist1)=ca.dist_centerline(oldcenterline,oldimg)
                    oldcent=dist1[-1]
                    (height1,dist1)=ca.dist_centerline(newcenterline,newimg)
                    newcent=dist1[-1]
                if newtime-oldtime!=0:
                    centerratio.append((newcent/oldcent)/(newtime-oldtime))
                    
                
    for direct in direct_list:
        ROIdic=np.load(direct+ROIdicname, allow_pickle=True)['arr_0'].item()
        maindict=np.load(direct+dicname, allow_pickle=True)['arr_0'].item()
        masklist=np.load(direct+masklistname, allow_pickle=True)['arr_0']
        for ROI in ROIdic.keys():
            if len(ROIdic[ROI]['Mask IDs'])>5:
                oldfile=masklist[ROIdic[ROI]['Mask IDs'][0]][2]
                
                oldindex=masklist[ROIdic[ROI]['Mask IDs'][0]][3]
                oldtime=maindict[oldfile]['time']
                
                oldsurf=maindict[oldfile]['area'][oldindex-1]
                
                
                newfile=masklist[ROIdic[ROI]['Mask IDs'][-1]][2]
                newindex=masklist[ROIdic[ROI]['Mask IDs'][-1]][3]
                newtime=maindict[newfile]['time']
                newsurf=maindict[newfile]['area'][newindex-1]
                if newtime-oldtime>0:
                    surfratio.append((newsurf/oldsurf)/(newtime-oldtime))
                if newtime-oldtime<0:
                    print(newtime,oldtime)
                    print(direct)
                    
                    
    plotstat(centerratio ,'centerline growth rate')
    plotstat(surfratio ,'surface growth rate')
    

        

    
        

if __name__ == "__main__":
    
    # whole_ROI,roots_only,leaf_only, non_connected,number_division=classifying_ROI(directory_list,ROI_name)
    # print(len(whole_ROI),len(roots_only),len(leaf_only), len(non_connected))
    # print('number_division',number_division)
    totale_stats(data_set2,ROI_name,dic_name,list_name)
    stats_time_evol(data_set2,ROI_name,dic_name,list_name)