# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:48:04 2023

@author: shawn
"""

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image #dealing with .tif images


#%% Centerline completion helper functions ####################################

def get_neighbors(coords,coords_lst):
    neighborhood = coords + np.array([[-1, -1],[-1, 0],[-1, 1],[0, -1],[0, 1],[1, -1],[1, 0],[1, 1]]) # get the possible neighbors of the current node
    neighbors = [tuple(nn) for nn in neighborhood for cc in coords_lst if np.all(nn==cc)] # get a list of all the neighbors 
    return neighbors


def find_extremal_div_pts(coords_lst):
    ex_pts = []
    div_pts = []
    for coordinates in coords_lst:
        neighbors = get_neighbors(coordinates,coords_lst)
        if len(neighbors)==1:
            ex_pts.append(coordinates)
        elif len(neighbors)>2:
            div_pts.append(coordinates)
    return ex_pts, div_pts


def dda_line(end_segment, mask):
    dy, dx = end_segment[-1][0] - end_segment[0][0], end_segment[-1][1] - end_segment[0][1]
    
    if abs(dx)>=abs(dy):
        step = abs(dx)
    else:
        step = abs(dy)
        
    dx = dx/step
    dy = dy/step
    x = end_segment[-1][1]
    y = end_segment[-1][0]
    
    line = []
    while True:
        x = x + dx
        y = y + dy
        x_coord = int(np.round(x))
        y_coord = int(np.round(y))
        
        if x_coord>=mask.shape[1] or y_coord>=mask.shape[0] or x_coord<0 or y_coord<0:
            break
        elif mask[y_coord,x_coord]!=0:
            line.append([y_coord,x_coord])
        else:
            break
        
    return line


#%% Wrapper function (accepts dictionary or directory input argument) #########

# def complete_centerlines(main_dict=None, dic_dir=None):
#     """
#     Extend the midlines of binarized images in a dictionary of masks and centerlines.

#     Parameters:
#         main_dict (dict, optional): A dictionary with keys representing file names and subkeys 'masks' 
#                                    and 'centerlines'. The 'masks' key should contain 2D numpy arrays 
#                                    of binarized images with masks over the regions of interest (ROIs). 
#                                    The 'centerlines' key should contain lists of numpy arrays that 
#                                    represent the y-x- coordinates of pixels making up the midlines of 
#                                    the masks. Default is None.
#         dic_dir (str, optional): The directory path to the dictionary. Default is None.
#                                  The dictionary can be optionally loaded from this path.

#     Returns:
#         main_dict: A modified dictionary with extended midlines. The 'centerlines' subkey of each 
#                   file entry will contain the updated numpy arrays with additional coordinates.
#     """
    
#     if dic_dir:
#         main_dict = np.load(dic_dir + "Main_dictionnary.npz", allow_pickle=True)['arr_0'].item()
        
        
#     for file_name in list(main_dict.keys()):
        
#         print("file name: ", file_name)
    
#         masks_arr = np.copy(main_dict[file_name]['masks'])
#         centerlines_arr = main_dict[file_name]['centerlines']
    
#         for ff in np.unique(masks_arr)[:-1]:
            
#             # print("mask label: ",ff+1)
            
#             mask = np.copy(masks_arr)
#             mask[masks_arr==ff+1] = 255
#             mask[mask!=255] = 0
            
#             centerline = np.copy(centerlines_arr[ff])
            
#             if len(centerline)<5: # skip if the centerline is too small or absent
#                 # print("skipped mask: centerline too small or missing")
#                 continue
                
#             end_points = find_extremal_div_pts(centerline)[0]
#             end_points = [list(e)for e in end_points]
            
#             if len(end_points)!=2: # skip if there are fewer or more than 2 terminal pixels in the centerline
#                 # print("skipped mask: fewer than 2 (looped) or more than 2 (branched) centerline terminal pixels")
#                 continue
            
#             extended_centerline = np.copy(centerline)
#             centerline_copy = np.ndarray.tolist(np.copy(centerline))
            
#             for end in end_points:
#                 end_segment = [end]
#                 centerline_copy.remove(end)
                
#                 for ii in range(5):
#                     neighbor = get_neighbors(end,centerline_copy)
                    
#                     if len(neighbor)!=1: # stop adding coordinates to the end segment if discontinuous or branched centerline
#                         # print("end segment branched or broken")
#                         break
#                     else:
#                         neighbor = list(neighbor[0])
                    
#                     # print(neighbor)
#                     end_segment.append(neighbor)
#                     # print(end_segment)
#                     centerline_copy.remove(neighbor)
#                     end = neighbor
                    
#                 end_segment = end_segment[::-1]
                
#                 if len(end_segment)>1: # extrapolate the end line segment if a slope can be calculated
#                     line = dda_line(end_segment, mask)
#                 else:
#                     # print("end segment too short for extrapolation")
#                     break
                
#                 if len(line)>0: # extend the centerline if the extension exists
#                     extended_centerline = np.concatenate((extended_centerline, np.array(line)))
#                 # else: 
#                     # print("centerline not extended from one end")
            
#             # for cc in extended_centerline:
#             #     masks_arr[cc[0],cc[1]] = 0
            
#             centerlines_arr[ff] = extended_centerline
        
#     return main_dict


#%% Alternative wrapper function (accepts one centerline numpy array and corresponding binarized mask numpy array)

def complete_one_centerline(centerline, mask):
    
    """
    Extrapolates the end segments of the centerline to the boundary of the given mask.

    Parameters:
        centerline (numpy.ndarray): Array of y-x coordinates representing the midline of a masked cell.
        
        mask (numpy.ndarray): Binarized array with non-zero values indicating the cell mask coordinates.
                              All other mask labels and background pixels should be set to 0.

    Returns:
            numpy.ndarray: Extended centerline with end segments extrapolated to the mask boundary.
    """
    # if len(centerline)<5: # skip if the centerline is too small or absent
        # print("Warning: centerline too small or missing")
        
    end_points = find_extremal_div_pts(centerline)[0]
    end_points = [list(e)for e in end_points]
    
    # if len(end_points)!=2: # skip if there are fewer or more than 2 terminal pixels in the centerline
        # print("Warning: fewer than 2 (looped) or more than 2 (branched) centerline terminal pixels")
    
    extended_centerline = np.copy(centerline)
    
    for end in end_points:
        end_segment = [end]
        centerline_copy = np.ndarray.tolist(np.copy(centerline))
        centerline_copy.remove(end)
        
        for ii in range(5):
            neighbor = get_neighbors(end,centerline_copy)
            
            if len(neighbor)!=1: # stop adding coordinates to the end segment if discontinuous or branched centerline
                # print("Warning: end segment branched or broken")
                break
            else:
                neighbor = list(neighbor[0])
            
            # print(neighbor)
            end_segment.append(neighbor)
            # print(end_segment)
            centerline_copy.remove(neighbor)
            end = neighbor
            
        end_segment = end_segment[::-1]
        
        if len(end_segment)>1: # extrapolate the end line segment if a slope can be calculated
            line = dda_line(end_segment, mask)
        # else:
            # print("Warning: end segment too short for extrapolation")
            
        
        if len(line)>0: # extend the centerline if the extension exists
            extended_centerline = np.concatenate((extended_centerline, np.array(line)))
        # else: 
            # print("Warning: centerline not extended from one end")
    
    # for cc in extended_centerline:
    #     masks_arr[cc[0],cc[1]] = 0
    
    # Sort coordinates of extended centerline
    centerline_copy = np.copy(extended_centerline).tolist()
    terminal_coordinates = find_extremal_div_pts(centerline_copy)[0][0]
    centerline_copy.remove(terminal_coordinates)
    reordered = [terminal_coordinates]
    for cc in range(len(centerline_copy)):
        neighbor = list(get_neighbors(reordered[-1],centerline_copy)[0])
        # print(neighbor)
        reordered.append(neighbor)
        centerline_copy.remove(neighbor)
    
    extended_centerline = np.array(reordered)
    
    return extended_centerline


#%% Load dictionary for testing ###############################################

# dic_dir = "C:/Users/shawn/OneDrive/Desktop/temp_scripts/stiffness_test/03-09-2014/"
# main_dict = np.load(dic_dir + "Main_dictionnary.npz", allow_pickle=True)['arr_0'].item()


# dic_dir = "C:/Users/shawn/OneDrive/Desktop/temp_scripts/centerline_tests/delta_parB_03-02-2015/"
# main_dict = np.load(dic_dir + "Main_dictionnary.npz", allow_pickle=True)['arr_0'].item()


# dic_dir = "C:/Users/shawn/OneDrive/Desktop/temp_scripts/centerline_tests/delta_parB_15-11-2014/"
# main_dict = np.load(dic_dir + "Main_dictionnary.npz", allow_pickle=True)['arr_0'].item()

#%% Troubleshooting, try frames 46 and 47

# file_names = list(main_dict.keys())

# file = file_names[46]
# masks_arr = main_dict[file]['masks']
# mask_labels = (np.unique(masks_arr)+1).tolist()
# centerlines = main_dict[file]['centerlines']

# fig, ax = plt.subplots()
# ax.imshow(masks_arr, cmap='binary')

# test = np.copy(masks_arr)
# test[test!=0] = 255

# for ii in range(len(centerlines)):
#     mask = np.copy(masks_arr)
#     mask[mask!=ii+1] = 0
#     mask[mask==ii+1] = 255
    
#     centerline = centerlines[ii]    
#     outline = main_dict[file]['outlines'][ii]
    
#     extended_centerline = complete_one_centerline(centerline, mask)
#     x_coords = [cc[1] for cc in extended_centerline]
#     y_coords = [cc[0] for cc in extended_centerline]
#     # ax.plot(x_coords, y_coords, 'r.', markersize=5)
#     plt.plot(x_coords, y_coords)

# plt.show()

#     for cc in extended_centerline:
#         test[cc[0],cc[1]] = 128
    
#     for cc in centerline:
#         test[cc[0],cc[1]] = 0
        
#     for cc in outline:
#         test[cc[0],cc[1]] = 0
    
#     # Image.fromarray(mask.astype(np.uint8)).show()

# Image.fromarray(test.astype(np.uint8)).show()

# check_masks = np.copy(masks_arr)
# check_masks[masks_arr!=0] = 255
# Image.fromarray(check_masks.astype(np.uint8)).show()


#%% Test 3rd centerline (index 2) of file 11152317 of dataset "delta_parB/15-11-2014/"

# file = "11152317"
# masks_arr = main_dict[file]['masks']
# mask_labels = (np.unique(masks_arr)+1).tolist()
# centerlines = main_dict[file]['centerlines']

# fig, ax = plt.subplots()
# ax.imshow(masks_arr, cmap='binary')

# # test = np.copy(masks_arr)
# # test[test!=0] = 255

# for ii in range(0,3):
#     mask = np.copy(masks_arr)
#     mask[mask!=ii+1] = 0
#     mask[mask==ii+1] = 255
    
#     # centerline = main_dict[file]['centerlines'][label-1]
#     centerline = centerlines[ii]    
#     outline = main_dict[file]['outlines'][ii]
    
    
#     extended_centerline = complete_one_centerline(centerline, mask)
#     x_coords = [cc[1] for cc in extended_centerline]
#     y_coords = [cc[0] for cc in extended_centerline]
#     ax.plot(x_coords, y_coords, 'r.', markersize=5)

# plt.show()

