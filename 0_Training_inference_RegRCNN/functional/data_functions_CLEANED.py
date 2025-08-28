# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:25:15 2017

@author: Tiger
"""

""" Retrieves validation images
"""

import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from skimage import measure
from natsort import natsort_keygen, ns
import os
import pickle
import scipy.io as sio
from tifffile import imsave

import zipfile
import bz2

#from plot_functions_CLEANED import *
#from data_functions import *
#from post_process_functions import *
#from UNet import *

from skan import skeleton_to_csgraph
from skimage.morphology import skeletonize_3d, skeletonize
import skimage


""" checks if nested lists are empty """
def isListEmpty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(isListEmpty, inList) )
    return False # Not a list
                        
        
        
""" Create ball """
def create_cube_in_im(width, input_size, z_size):
    cube_in_middle = np.zeros([input_size,input_size, z_size])
    cube_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
    cube_in_middle_dil = dilate_by_cube_to_binary(cube_in_middle, width=width)
    center_cube = cube_in_middle_dil
    
    return center_cube


""" dilates image by a spherical ball of size radius """
def erode_by_ball_to_binary(input_im, radius):
     ball_obj = skimage.morphology.ball(radius=radius)
     input_im = skimage.morphology.erosion(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" dilates image by a spherical ball of size radius """
def dilate_by_ball_to_binary(input_im, radius):
     ball_obj = skimage.morphology.ball(radius=radius)
     input_im = skimage.morphology.dilation(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im




""" Use MATLAB dilation """
# import matlab.engine
# eng = matlab.engine.start_matlab()
# def dilate_by_ball_MATLAB(input_im, radius):
#       strel = eng.strel('sphere', matlab.double([1]))
#       #eng.getfield(disk, 'Neighborhood')
#       dilated = eng.imdilate(matlab.double(input_im.tolist()), strel)
#       py_arr = np.asarray(dilated)



""" dilates image by a spherical ball of size radius """
def dilate_by_disk_to_binary(input_im, radius):
     ball_obj = skimage.morphology.disk(radius=radius)
     input_im = skimage.morphology.dilation(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" dilates image by a cube of size width """
def dilate_by_cube_to_binary(input_im, width):
     cube_obj = skimage.morphology.cube(width=width)
     input_im = skimage.morphology.dilation(input_im, selem=cube_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" erodes image by a cube of size width """
def erode_by_cube_to_binary(input_im, width):
     cube_obj = skimage.morphology.cube(width=width)
     input_im = skimage.morphology.erosion(input_im, selem=cube_obj)  
     input_im[input_im > 0] = 1
     return input_im


""" Applies CLAHE to a 2D image """           
# def apply_clahe_by_slice(crop, depth):
#      clahe_adjusted_crop = np.zeros(np.shape(crop))
#      for slice_idx in range(depth):
#           slice_crop = np.asarray(crop[:, :, slice_idx], dtype=np.uint8)
#           adjusted = equalize_adapthist(slice_crop, kernel_size=None, clip_limit=0.01, nbins=256)
#           clahe_adjusted_crop[:, :, slice_idx] = adjusted
                 
#      crop = clahe_adjusted_crop * 255
#      return crop

""" Take input bw image and returns coordinates and degrees pixel map, where
         degree == # of pixels in nearby CC space
                 more than 3 means branchpoint
                 == 2 means skeleton normal point
                 == 1 means endpoint
         coordinates == z,x,y coords of the full skeleton object
         
    *** works for 2D and 3D inputs ***
"""

def bw_skel_and_analyze(bw):
     if bw.ndim == 3:
          skeleton = skeletonize_3d(bw)
     elif bw.ndim == 2:
          skeleton = skeletonize(bw)
     skeleton[skeleton > 0] = 1
    
     
     if skeleton.any() and np.count_nonzero(skeleton) > 1:
          try:
               pixel_graph, coordinates, degrees = skeleton_to_csgraph(skeleton)
               coordinates = coordinates[0::]   ### get rid of zero at beginning
          except:
               pixel_graph = np.zeros(np.shape(skeleton))
               coordinates = []
               degrees = np.zeros(np.shape(skeleton))               
     else:
          pixel_graph = np.zeros(np.shape(skeleton))
          coordinates = []
          degrees = np.zeros(np.shape(skeleton))
          
     return pixel_graph, degrees, coordinates



""" Sort skeleton as graph into ordered list of coordinates starting from start point

    Returns:
        - ordered coordinates
        - coordinates of branch points
        - coordinates of end points
"""
from scipy.sparse.csgraph import shortest_path, breadth_first_tree
from tsp_solver.greedy import solve_tsp
from sklearn.neighbors import NearestNeighbors


import networkx as nx



""" order any list of coordinates based on a starting point """
def order_coords_from_start(coords, start):
    """ Move start coord to beginning """
    idx_start = np.where((coords == (start)).all(axis=1))[0][0]
    coords = np.delete(coords, idx_start, axis=0)
    coords = np.vstack((start, coords))    
            
    """ Generate sparse matrix """
    clf = NearestNeighbors(n_neighbors=2).fit(coords)
    G = clf.kneighbors_graph()


    """ Generate ordered nodes so can figure out order to visit branch and endpoints """    
    T = nx.from_scipy_sparse_matrix(G)

    
    """ Then TREEIFY by looping through list and visiting each coord until hits branch or end
        
            and add that segment into tree!!!
            - ***Maybe make this separate function???
        """

    tree_T = nx.dfs_tree(T, source=0)
    tree_order = list(tree_T.edges())
    
    ordered, discrete_segs = recurse_NX_tree(coords, tree_order, cur_idx=0, parent_idx=0, ordered=[], discrete_segs=[], cur_ordered=[])
    ordered = np.vstack(ordered)     
    
    return ordered, tree_order, discrete_segs
    
    
""" Take in a skeleton and transform it into ordered list + ordered tree """
def order_skel_graph(degrees, start=[], end=[]):

    bw = np.copy(degrees); bw[bw > 0] = 1
    cc_be = measure.regionprops(bw)
    be_coords = []
    for obj in cc_be: be_coords.append(obj['coords'])
    all_coords = np.vstack(be_coords)
    

    """ Get coordinates of branch and endpoints """
    only_segments = np.copy(degrees); only_segments[only_segments != 2] = 0
    only_branches = np.copy(degrees); only_branches[only_branches == 2] = 0   
    
    ### convert branch and endpoints into a list with +/- neihgbourhood values
    #labels = measure.label(only_branches)
    cc_be = measure.regionprops(only_branches)
    be_coords = []
    for obj in cc_be: be_coords.append(obj['coords'])
    
    
    """ Generate ordered coords and tree """
    ordered, tree_order, discrete_segs = order_coords_from_start(all_coords, start)


    return ordered, discrete_segs, be_coords
    
    
    """ Also delete any branches that do NOT end in an endpoint AND have no children (i.e. loops onto itself) """

""" Walk through NX tree and get ordered coordinates """    
def recurse_NX_tree(all_coords, tree_order, cur_idx, parent_idx, ordered, discrete_segs, cur_ordered):

    parent = tree_order[cur_idx][0]
    child = tree_order[cur_idx][1]
    
    if parent == 0:
        ordered.append(all_coords[parent])
        cur_ordered.append(all_coords[parent])
    
    ordered.append(all_coords[child])
    cur_ordered.append(all_coords[child])
    
    all_indices = np.vstack(tree_order)[:, 0]
    idx_child = np.where(all_indices == child)[0]
    
    
    ### if no more children, then also end
    #print(ordered)
    if len(idx_child) == 0:
        discrete_segs.append(cur_ordered)   ### make sure have end_be_coord
        cur_ordered = []
        return ordered, discrete_segs
    
    if len(idx_child) > 1:
        discrete_segs.append(cur_ordered)            

    for idx in idx_child:
        if len(idx_child) > 1:
            #print(cur_ordered)
            cur_ordered = []
            cur_ordered = list([all_coords[child]])         ### start with parent node so can have start_be_index in list!!!
            
            
        coords, discrete_segs = recurse_NX_tree(all_coords, tree_order, cur_idx=idx, parent_idx=idx, ordered=ordered, discrete_segs=discrete_segs, cur_ordered=cur_ordered)
        #ordered.append(coords)
    
    return ordered, discrete_segs
    


""" Treeify the discretized segments by adding them to tree df """
def elim_loops(discrete_segs, tree_idx, disc_idx, parent, be_coords=[], cleaned_segs=[], start_tree=0):

    cur_seg = discrete_segs[disc_idx]
    cur_seg = np.vstack(cur_seg)
    
    start = cur_seg[0]
    coords = cur_seg
    end = cur_seg[-1]
    
    #print(len(cleaned_segs))
    
                                  
    ### find next places to go by looking at which next segments starting points match current segment end point
    children = []
    for idx, seg in enumerate(discrete_segs):
        if idx == disc_idx:
            continue ### skip over current segment
        start_check = seg[0]
        #print(start_check)
        if (start_check == end).all():
            children.append(idx)
            
    """ DON'T add if there are NO children AND the end coord is not contained in the list be_coords
            this means that this is a loop that comes back onto itself
    """
    
    if len(children) == 0 and len(be_coords) > 0:
        end_points = be_coords[0];
        match = 0;
        for row in end_points:
            if (row == end).all():
                match = 1
                
        for row in end_points:
            if (np.vstack([coords, end]) == row).all(-1).any():
               match = 1 
                        
    else: match = 1
    

    
    if match:     
        ### go to all children
        cleaned_segs.append(cur_seg)
        
        for child in children:
            cleaned_segs = elim_loops(discrete_segs, tree_idx=tree_idx, disc_idx=child, parent=child, be_coords=be_coords, cleaned_segs=cleaned_segs)
            
            #cleaned_segs.append(child_segs)
            
    return cleaned_segs
    
    
    
""" Treeify the discretized segments by adding them to tree df """
def treeify_nx(tree, discrete_segs, tree_idx, disc_idx, parent, be_coords=[], start_tree=0):
    
    if disc_idx == -1:
        return tree
    else:
        cur_seg = discrete_segs[disc_idx]
        cur_seg = np.vstack(cur_seg)
        
        start = cur_seg[0]
        coords = cur_seg
        end = cur_seg[-1]
                     
 
                        
        ### find next places to go by looking at which next segments starting points match current segment end point
        children = []
        for idx, seg in enumerate(discrete_segs):
            if idx == disc_idx:
                continue ### skip over current segment
            start_check = seg[0]
            #print(start_check)
            if (start_check == end).all():
                children.append(idx)

        num_missing = []
        if len(tree) == 0:
            cur_idx = 0   ### if starting tree from nothing!
                
       
        elif len(tree) > 1:
            
            ### First check if there is a missing value, because fill that first, otherwise, just do max + 1
            #print(tree.cur_idx)
            lst = np.asarray(tree.cur_idx)
            #print(cur_idx)
            num_missing = [x for x in range(int(lst[0]), int(lst[-1]+1)) if x not in lst] 
            if len(num_missing) > 0:
                cur_idx = num_missing[0]
            else:                    
                cur_idx = np.max(tree.cur_idx[:]) + 1; 
        else:
            cur_idx = np.max(tree.cur_idx[:]) + 1; 

                                       

        #print(children)
        ### add to tree
        if (len(children) > 0 and start_tree) or len(tree) == 0: child_vals = np.add(children,  0).tolist();
        
        elif len(children) > 0 and len(num_missing) > 0:  child_vals = np.add(children, int(np.asarray(tree.cur_idx)[-1])).tolist()
        elif len(children) > 0: child_vals = np.add(children, int(np.asarray(tree.cur_idx)[-1])).tolist()
        
        
        else: child_vals = []
            
        
        """ DON'T add if there are NO children AND the end coord is not contained in the list be_coords
                this means that this is a loop that comes back onto itself
        """
        
        if len(children) == 0 and len(be_coords) > 0:
            end_points = be_coords[0];
            match = 0;
            for row in end_points:
                if (row == end).all():
                    match = 1
                    
            for row in end_points:
                if (np.vstack([coords, end]) == row).all(-1).any():
                   match = 1 
                    
            # if not match:
            #     try:
            #         ### must also remove child from parent
            #         childs_to_change = tree.child[parent]
            #         print(cur_idx)
            #         childs_to_change.remove(cur_idx)
                    
            #         ### shift all indices above this index down by 1 because something has been deleted
            #         childs_to_change = np.asarray(childs_to_change)
            #         childs_to_change[np.where(childs_to_change > cur_idx)[0]] = childs_to_change[np.where(childs_to_change > cur_idx)[0]] - 1
            #         tree.child[parent] = childs_to_change
            #     except:
            #         print('already deleted?')
                                            
        else: match = 1
                    
            
        
        if match:
        
            new_node = {'coords': coords, 'parent': parent, 'child': child_vals, 'depth': 0, 
                        'cur_idx': int(cur_idx), 'start_be_coord': start, 'end_be_coord': end, 'visited': np.nan}
     
            if len(children) > 0:
                new_node['visited'] = 1
        
            ### if it's a deleted node, then add back into the location it was deleted from!!!
            if len(num_missing) > 0:
                tree.loc[cur_idx] = new_node
                tree = tree.sort_index()
                ### else, add it to the end of the list
            else:
                tree = tree.append(new_node, ignore_index=True)
            
               
            ### go to all children
            for child in children:
                tree = treeify_nx(tree, discrete_segs, tree_idx=tree_idx, disc_idx=child, parent=cur_idx, be_coords=be_coords)
                
                
                
                #print(child)
            
        return tree
        
        
        
    
    
    






""" removes detections on the very edges of the image """
def clean_edges(im, extra_z=1, extra_xy=3, skip_top=0):
     im_size = np.shape(im);
     w = im_size[1];  h = im_size[2]; depth = im_size[0];
     labelled = measure.label(im)
     cc_coloc = measure.regionprops(labelled)
    
     cleaned_im = np.zeros(np.shape(im))
     for obj in cc_coloc:
         coords = obj['coords']
         
         bool_edge = 0
         for c in coords:
              if ((c[0] <= 0 + extra_z and not skip_top) or c[0] >= depth - extra_z):
                   bool_edge = 1
                   break;
              if (c[1] <= 0 + extra_xy or c[1] >= w - extra_xy):
                   bool_edge = 1
                   break;                                       
              if (c[2] <= 0 + extra_xy or c[2] >= h - extra_xy):
                   bool_edge = 1
                   break;                                        

         if not bool_edge:
              for obj_idx in range(len(coords)):
                   cleaned_im[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1

     return cleaned_im                           
 

def find_TP_FP_FN_from_im(seg_train, truth_im):

     coloc = seg_train + truth_im
     bw_coloc = coloc > 0
     labelled = measure.label(truth_im)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     
     true_positive = np.zeros(np.shape(coloc))
     TP_count = 0;
     FN_count = 0;
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
          if max_val > 1:
               TP_count += 1
               #for obj_idx in range(len(coords)):
               #     true_positive[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1
          else:
               FN_count += 1
 
     
     FP_count = 0;
     labelled = measure.label(bw_coloc)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
          if max_val == 1:
               FP_count += 1 
              
     return TP_count, FN_count, FP_count

           

def find_TP_FP_FN_from_seg(segmentation, truth_im, size_limit=0):
     seg = segmentation      
     true = truth_im  
     
     """ Also remove tiny objects from Truth due to error in cropping """
     labelled = measure.label(true)
     cc_coloc = measure.regionprops(labelled)
     
     cleaned_truth = np.zeros(np.shape(true))
     for obj in cc_coloc:
          coords = obj['coords']
          
          # can also skip by size limit          
          if len(coords) > 10:
               for obj_idx in range(len(coords)):
                    cleaned_truth[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1
    
     
     """ Find matched """
     coloc = seg + true
     bw_coloc = coloc > 0
     labelled = measure.label(true)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     
     true_positive = np.zeros(np.shape(coloc))
     TP_count = 0;
     FN_count = 0;
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
          
          # can also skip by size limit          
          if max_val > 1 and len(coords) > size_limit:
               TP_count += 1
          else:
               FN_count += 1
 
     
     FP_count = 0;
     labelled = measure.label(seg)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     cleaned_seg = np.zeros(np.shape(seg))
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
     
          # can also skip by size limit
          if  len(coords) < size_limit:
               continue;
          else:
               for obj_idx in range(len(coords)):
                    cleaned_seg[coords[obj_idx, 0], coords[obj_idx, 1], coords[obj_idx, 2]] = 1
          
          if max_val == 1:
               FP_count += 1 
              
     return TP_count, FN_count, FP_count, cleaned_truth, cleaned_seg





""" Convert voxel list to array """
def convert_vox_to_matrix(voxel_idx, zero_matrix):
    for row in voxel_idx:
        #print(row)
        zero_matrix[(row[0], row[1], row[2])] = 1
    return zero_matrix



""" converts a matrix into a multipage tiff to save!!! """
def convert_matrix_to_multipage_tiff(matrix):
    rolled = np.rollaxis(matrix, 2, 0).shape  # changes axis to be correct sizes
    tiff_image = np.zeros((rolled), 'uint8')
    for i in range(len(tiff_image)):
        tiff_image[i, :, :] = matrix[:, :, i]
        
    return tiff_image


""" Saving the objects """
def save_pkl(obj_save, s_path, name):
    with open(s_path + name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([obj_save], f)

"""Getting back the objects"""
def load_pkl(s_path, name):
    with open(s_path + name, 'rb') as f:  # Python 3: open(..., 'rb')
      loaded = pickle.load(f)
      obj_loaded = loaded[0]
      return obj_loaded



"""
    To normalize by the mean and std
"""
def normalize_im(im, mean_arr, std_arr):
    normalized = (im - mean_arr)/std_arr 
    return normalized        
    




