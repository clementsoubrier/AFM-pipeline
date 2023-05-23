#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:15:33 2023

@author: c.soubrier
"""



from sklearn import manifold
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as col

A=np.load('centerline_analysis_result/centerline_list_3.npy')
B=np.load('centerline_analysis_result/distance_matrix_3.npy')
C=np.load('centerline_analysis_result/inversion_matrix_3.npy')
D=np.load('centerline_analysis_result/delta_matrix_3.npy')


result_path='Height/Dic_dir/'

dic_name='Main_dictionnary.npy'

#220-260
debut=200
fin=-1
B=B[debut:fin,debut:fin]
print(len(B))
colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]


plt.imshow(B,'Greys_r')
plt.show()

mds = manifold.MDS(n_components=2, random_state = 1, dissimilarity="precomputed")
pos = mds.fit(B).embedding_


fig = plt.figure(figsize=(20, 12))
plt.title("MDS of the whole dataset, with lineage")

# plt.xlim(-200, 100)
# plt.ylim(-70, 100)
for i in range(len(pos)):
    infos=A[i+debut]
    if i==0:
        dic=np.load(infos[1]+result_path+dic_name, allow_pickle=True).item()
        mask_list=np.load(infos[1]+result_path+'masks_list.npy', allow_pickle=True)
    adresse,masknumber=(mask_list[int(infos[2])])[2:]
    col=int(dic[adresse]['basic_graph_values'][masknumber-1])
    plt.scatter(pos[i, 0], pos[i, 1],c=[np.array(colormask[col%len(colormask)-1])/255])
    

plt.show()
'''
fig = plt.figure(figsize=(20, 12))
plt.title("Clustering of the MDS")
# plt.xlim(-200, 100)
# plt.ylim(-70, 100)
kmeans = cluster.KMeans(n_clusters=6, random_state=0, n_init=100).fit(pos)#
for i in range(len(pos)):
    col= kmeans.labels_[i]
    plt.scatter(pos[i, 0], pos[i, 1],c=[np.array(colormask[col%len(colormask)-1])/255])
plt.show()

finlist=[]
for i in range(len(B)):
    newlist=B[i]
    sublist=[]
    for j in range(len(B)):
        if j!=14 and j!=13:
            sublist.append(newlist[j])
    if i!=14 and i!=13:
        finlist.append(sublist)

B=np.array(finlist)            


mds = manifold.MDS(n_components=2, random_state = 1, dissimilarity="precomputed")
pos = mds.fit(B).embedding_


fig = plt.figure(figsize=(20, 12))
plt.title("Second MDS, without outliers")

# plt.xlim(-200, 100)
# plt.ylim(-70, 100)
for i in range(len(pos)):
    k=0
    if i>=13:
        k=2
    infos=A[i+debut+k]
    if i==0:
        dic=np.load(infos[0], allow_pickle=True).item()
    adresse=infos[1]
    print(infos[2])
    col=int(dic[adresse]['basic_graph_values'][int(infos[2])])
    plt.scatter(pos[i, 0], pos[i, 1],c=[np.array(colormask[col%len(colormask)-1])/255])
    

plt.show()

fig = plt.figure(figsize=(20, 12))
plt.title("Clustering of the second MDS")
# plt.xlim(-200, 100)
# plt.ylim(-70, 100)
kmeans = cluster.KMeans(n_clusters=6, random_state=0, n_init=100).fit(pos)#
for i in range(len(pos)):
    col= kmeans.labels_[i]
    plt.scatter(pos[i, 0], pos[i, 1],c=[np.array(colormask[col%len(colormask)-1])/255])
plt.show()
'''
'''
fig = plt.figure(figsize=(20, 12))
plt.title("MDS of the test dataset 2")
# plt.xlim(-500, 500)
# plt.ylim(-600, 300)

for i in range(len(pos)//5-1):#
    infos=A[i+3*len(pos)//5]
    adresse=infos[1]
    print(infos[2])
    col=int(dic[adresse]['basic_graph_values'][int(infos[2])])
    plt.scatter(pos[i+3*len(pos)//5, 0], pos[i+3*len(pos)//5, 1],c=[np.array(colormask[col%len(colormask)-1])/255])
    

plt.show()



fig = plt.figure(figsize=(20, 12))
plt.title("MDS of the test dataset time")

# plt.xlim(-500, 500)
# plt.ylim(-600, 300)
time=[]
for i in range(len(pos)):
    infos=A[i]
    if i==0:
        dic=np.load(infos[0], allow_pickle=True).item()
    adresse=infos[1]
    time.append(dic[adresse]['time']+1)
    maxtime=max(time)

for i in range(len(pos)):
    infos=A[i]
    if i==0:
        dic=np.load(infos[0], allow_pickle=True).item()
    adresse=infos[1]
    col1=0.1+0.8*time[i]/maxtime
    #col2=int(dic[adresse]['basic_graph_values'][int(infos[2])])
    #plt.scatter(pos[i, 0], pos[i, 1],c=[np.array(colormask[col2%len(colormask)-1])/255*col1])
    plt.scatter(pos[i, 0], pos[i, 1],c=str(col1))
    

plt.show()
'''