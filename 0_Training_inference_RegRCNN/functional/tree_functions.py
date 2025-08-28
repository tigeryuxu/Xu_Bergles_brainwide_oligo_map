#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:26:11 2020

@author: user
"""


import numpy as np
from functional.matlab_crop_function import *
#from functional.init_seg_functions import *

#from functional.plot_functions_CLEANED import *
#from functional.data_functions_CLEANED import *
#from functional.skel_and_graph_ordering_functions import *
import pandas as pd

from scipy.sparse.csgraph import depth_first_order
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from skimage.draw import line_nd
from skimage import measure

from scipy.spatial import distance 




""" Bridge adjacent end points within radius of 2 pixels """
def bridge_end_points(output_PYTORCH, bridge_radius=2):
    ### (1) Find cc of every object in the current crop
    labelled = measure.label(output_PYTORCH)
    cc = measure.regionprops(labelled)
  
    all_seg = []
    for cur in cc:
        cur_seg = {
                "coords": cur['coords'],
                "center_be": [],
                "expand_be": [],
                "bridges": [],
            }
        all_seg.append(cur_seg)
            
  
    ### (2) Find the end points AND(???) branchpoints
    pixel_graph, degrees, coordinates = bw_skel_and_analyze(output_PYTORCH)
    be_points = np.copy(degrees); be_points[be_points == 2] = 0; be_points[be_points > 0] = 1;
    
    labelled = measure.label(be_points)
    cc_be = measure.regionprops(labelled)
    
    ### (3) get pixel indices of each be AND get the expanded version of the be neighborhood as well 
    # match the end point to the list of coords
    for seg in all_seg:
        cur_seg = seg['coords']
        for be in cc_be:
            cur_be = be['coords']
            if (cur_seg[:, None] == cur_be).all(-1).any():
                
                seg["center_be"].append(cur_be)
                
                ### expand the cur_be
                neighborhood_be = expand_coord_to_neighborhood(cur_be, lower=bridge_radius, upper=bridge_radius + 1)
                if len(neighborhood_be) > 0:
                    neighborhood_be = np.vstack(neighborhood_be)
                seg["expand_be"].append(neighborhood_be)
                
                
    ### (4) loop through each cc and see if be neighborhood hits nearby cc EXCLUDING itself
    ### if it hits, use line_nd to make connection
    empty = np.zeros(np.shape(output_PYTORCH))
    for cur_seg, cur_idx in zip(all_seg, range(len(all_seg))):
        cur_expand = cur_seg['expand_be']
        if len(cur_expand) > 0:
            for cur_ex, idx_outer in zip(cur_expand, range(len(cur_expand))):  # loop through each be of current seg
                 for next_seg, next_idx in zip(all_seg, range(len(all_seg))):  # loop through all other segs
                     if cur_idx == next_idx:
                         continue;   ### don't try to match with self
                     
                     ### (a) try to find an expanded neighborhood that matches
                     match = 0
                     next_expand = next_seg['expand_be']
                     if len(next_expand) > 0:
                         all_lines = []
                         dist_lines = []
                         for be_ex, idx_inner in zip(next_expand, range(len(next_expand))): # loop through each be of next seg             
                               if (cur_ex[:, None] == be_ex).all(-1).any():  
                                   
                                   next_be = next_seg['center_be'][idx_inner][0]
                                   cur_be = cur_seg['center_be'][idx_outer][0]
                                   
                                   ### DRAW LINE
                                   line_coords = line_nd(cur_be, next_be, endpoint=False)
                                   line_coords = np.transpose(line_coords)
                                   
                                   all_lines.append(line_coords)
                                   dist_lines.append(len(line_coords))
                                   
                         ### ONLY ADD THE SHORTEST LINE:
                         if len(dist_lines) > 0:
                             cur_seg['bridges'].append(all_lines[np.argmin(dist_lines)])
                             match = 1
                         #print('bridge')

            
                     ### (b) then try to find coords that match ==> ***IF MATCHED, find CLOSEST
                     if not match:
                          next_coords = next_seg['coords']
                          if len(next_coords) > 0 and (cur_ex[:, None] == next_coords).all(-1).any():
                              cur_be = cur_seg['center_be'][idx_outer][0]
                              # find distance to all coords
                             
                              cur = np.transpose(np.vstack(cur_be))
                              dist = distance.cdist(cur, next_coords)
                              min_idx = np.argmin(dist)
                             
                              closest_point = next_coords[min_idx]
                            
                              ### DRAW LINE
                              line_coords = line_nd(cur_be, closest_point, endpoint=False)
                              line_coords = np.transpose(line_coords)
                              cur_seg['bridges'].append(line_coords)     
                             
                              print('body bridge')
                              #print(cur_be)
                             
    ### debug: ensure proper points inserted
    """ get output image """
    output = np.zeros(np.shape(output_PYTORCH))
    for seg, idx in zip(all_seg, range(len(all_seg))):
         cur_expand = seg['bridges']
         if len(cur_expand) > 0:
             cur_expand = np.vstack(cur_expand)
             output[cur_expand[:, 0], cur_expand[:, 1], cur_expand[:, 2]] = 5
  
         cur_seg = seg['coords']
         if len(cur_seg) > 0:
              cur_seg = np.vstack(cur_seg)
              output[cur_seg[:, 0], cur_seg[:, 1], cur_seg[:, 2]] = idx + 1
            
  

    non_bin_output = np.copy(output)
    output[output > 0] = 1
    
    
    return output, non_bin_output

""" Given coords of shape x, y, z in a cropped image, scales back to size in full size image """
def scale_coords_of_crop_to_full(coords, box_xyz, box_over):
        coords[:, 0] = np.round(coords[:, 0]) + (box_xyz[0] - box_over[0])   # SCALING the ep_center
        coords[:, 1] = np.round(coords[:, 1]) + (box_xyz[2] - box_over[2])
        coords[:, 2] = np.round(coords[:, 2]) + (box_xyz[4] - box_over[4])
        scaled = coords
        return scaled  

""" Given coord of shape x, y, z in a cropped image, scales back to size in full size image """
def scale_coord_to_full(coord, box_xyz, box_over):
        coord[0] = np.round(coord[0]) + (box_xyz[0] - box_over[0])   # SCALING the ep_center
        coord[1] = np.round(coord[1]) + (box_xyz[2] - box_over[2])
        coord[2] = np.round(coord[2]) + (box_xyz[4] - box_over[4])
        scaled = coord
        return scaled  



                  

""" Connect any coords that are > 1 pixel away """
def connect_nearby_px(coords):
    
    """ must only look at UNIQUE elements """
    coords = np.unique(coords, axis=0)
    clf = NearestNeighbors(n_neighbors=3).fit(coords)
    distances, indices = clf.kneighbors(coords)
    
    ### need to connect FOR 2nd NEAREST NEIGHBOR
    ind_to_c = np.where(distances[:, 1] >= 2)[0]
    
    full_coords = []
    full_coords.append(coords)
    for ind in ind_to_c:
        start = indices[ind][0]
        end = indices[ind][1]
        
        line_coords = line_nd(coords[start], coords[end], endpoint=False)
        line_coords = np.transpose(line_coords)  
        
        full_coords.append(line_coords[1:len(line_coords)])   ### don't reappend the starting coordinate
    
    ### NEED TO CONNECT FOR 3rd NEAREST NEIGHBOR
    
    ind_to_c = np.where(distances[:, 2] >= 2)[0]
    
    #full_coords = []
    #full_coords.append(coords)
    for ind in ind_to_c:
        start = indices[ind][0]
        end = indices[ind][2]
        
        line_coords = line_nd(coords[start], coords[end], endpoint=False)
        line_coords = np.transpose(line_coords)  
        
        full_coords.append(line_coords[1:len(line_coords)])   ### don't reappend the starting coordinate
        
    
        
    full_coords = np.vstack(full_coords)
    
    
    return full_coords



""" Returns the coordinates of the parents of the current starting index """
def get_parent_nodes(tree, start_ind, num_parents, parent_coords):
    
    if num_parents == 0 or start_ind == -1:
        return parent_coords
    
    else:
        parent_ind = tree.parent[start_ind] 
        
        if parent_ind == -1:
            #print("hit bottom of tree")
            return parent_coords
      
        
        parent_coords = get_parent_nodes(tree, start_ind=parent_ind, num_parents=num_parents - 1, parent_coords=parent_coords)


        ### HACK: do this instead if neighborhood is just single coordinate
        if len(tree.start_be_coord[parent_ind]) == 3:
            #print('yeet')
            parent_coords.append(tree.start_be_coord[parent_ind])
            parent_coords.append(tree.coords[parent_ind])
            parent_coords.append(tree.end_be_coord[parent_ind])
    

                    
        else:
            parent_coords.append(tree.start_be_coord[parent_ind][math.floor(len(tree.start_be_coord[parent_ind])/2)])
            parent_coords.append(tree.coords[parent_ind])
        
            if not np.isnan(tree.end_be_coord[parent_ind]).any():
                parent_coords.append(tree.end_be_coord[parent_ind][math.floor(len(tree.end_be_coord[parent_ind])/2)])
    

        #print(parent_ind)
        return parent_coords







""" Gets image of tree where is lowest values near cell body """
def get_rainbow_im(tree, input_im, lower_exp=2, upper_exp=3):

    #tree = all_trees[0].copy()
    reset_indices = np.where(tree.visited == 1)[0]
    tree.visited[reset_indices] = np.nan
    
    
    shortest_order = []
    while np.asarray(tree.visited.isnull()).any():
        unvisited_indices = np.where(tree.visited.isnull() == True)[0]   ### Get index of unvisited nodes       
        
        all_lengths = []
        for ind in unvisited_indices:
            parent_coords = get_parent_nodes(tree, ind, num_parents=10000, parent_coords = [])

            if len(parent_coords) > 0:
                parent_coords = np.vstack(parent_coords)

            all_lengths.append(len(parent_coords))
        node_idx = unvisited_indices[np.argmin(all_lengths)]
        print(node_idx)
        shortest_order.append(node_idx)
        
        tree.visited[node_idx] = 1
        
        
    rainbow_im = np.zeros(np.shape(input_im))
    
    for node_id in shortest_order:
        
        coords = tree.coords[node_id]
        ind = np.asarray(tree.cur_idx[node_id])
        start_be = np.asarray(tree.start_be_coord[node_id])
    
        end_be = np.asarray(tree.end_be_coord[node_id])
    
        expanded = np.asarray(expand_coord_to_neighborhood(coords, lower=lower_exp, upper=upper_exp))
        
        expanded[:, 0] = np.clip(expanded[:, 0], a_min=0, a_max=np.shape(input_im)[0] - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], a_min=0, a_max=np.shape(input_im)[1] - 1)
        expanded[:, 2] = np.clip(expanded[:, 2], a_min=0, a_max=np.shape(input_im)[2] - 1)

        rainbow_im[expanded[:, 0], expanded[:, 1], expanded[:, 2]] = ind + 1 # because dont want non-zero

        

        expanded = np.asarray(expand_coord_to_neighborhood([start_be], lower=lower_exp + 1, upper=upper_exp + 1))
        expanded[:, 0] = np.clip(expanded[:, 0], a_min=0, a_max=np.shape(input_im)[0] - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], a_min=0, a_max=np.shape(input_im)[1] - 1)
        expanded[:, 2] = np.clip(expanded[:, 2], a_min=0, a_max=np.shape(input_im)[2] - 1)

        rainbow_im[expanded[:, 0], expanded[:, 1], expanded[:, 2]] = 255           

        expanded = np.asarray(expand_coord_to_neighborhood([end_be], lower=lower_exp + 1, upper=upper_exp + 1))
        expanded[:, 0] = np.clip(expanded[:, 0], a_min=0, a_max=np.shape(input_im)[0] - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], a_min=0, a_max=np.shape(input_im)[1] - 1)
        expanded[:, 2] = np.clip(expanded[:, 2], a_min=0, a_max=np.shape(input_im)[2] - 1)
        
        
        
        rainbow_im[expanded[:, 0], expanded[:, 1], expanded[:, 2]] = 255          

    return rainbow_im, shortest_order




""" Convert list into tree in pandas dataframe """
def get_tree_from_im_list(root, input_im, width_tmp, height_tmp, depth_tmp, all_coords_root):
    columns = {'coords', 'parent', 'child', 'depth', 'start_be_coord', 'end_be_coord', 'cur_idx', 'visited'}
    tree_df = pd.DataFrame(columns=columns)
    

    """ add the root to an empty image and find it's skeleton + branchpoints + edges ==> maybe needs a bit of smoothing as well??? """
    empty_root_im = np.zeros(np.shape(input_im))
    for idx_trace in range(len(root)):
          empty_root_im[root[idx_trace][0], root[idx_trace][1], root[idx_trace][2]] = 1        
          if idx_trace == int(len(root)/2):
               centroid = [root[idx_trace][0], root[idx_trace][1], root[idx_trace][2]]    
               
    """ Crop for speed skeletonization """    
    x = int(centroid[0]); y = int(centroid[1]); z = int(centroid[2])
    crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(empty_root_im, y, x, z, 
                                                                                crop_size=250, z_size=100, height=height_tmp, width=width_tmp, depth=depth_tmp)
     
    """ add endpoints to a new list of points to visit (seed_idx) """
    pixel_graph, degrees_small, coordinates = bw_skel_and_analyze(crop)
    degrees_full_size = np.zeros(np.shape(input_im))
    degrees_full_size[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = degrees_small
    degrees = degrees_full_size
    
    end_points = np.copy(degrees); end_points[end_points != 1] = 0
    branch_points = np.copy(degrees); branch_points[branch_points <= 2] = 0  ### ANYTHING ABOVE 2 is a branchpoint!
                                ### number of pixel value in degrees is connectivity!!!
                            
    coords_end_points = np.transpose(np.nonzero(end_points))



    """ HACK: find point closest to interactive scroller """
    # degrees_small[degrees_small > 0] = 1
    # only_colocalized_mask, overall_coord = GUI_cell_selector(degrees_small, crop_size=100, z_size=30,
    #                                                         height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_tmp, thresh=0)
    
    # overall_coord = scale_coords_of_crop_to_full(np.transpose(np.vstack(overall_coord)), box_x_min, box_y_min, box_z_min)


    """ To find root ==> is side closest to middle of cell...??? """
    dist_to_root = []
    coord_root = [0, 0, 0]
    for check_coord in all_coords_root:
        expanded = expand_coord_to_neighborhood(check_coord, lower=1, upper=2)
        if (np.vstack(expanded)[:, None] == coords_end_points).all(-1).any():            
            #coord_root = check_coord
            
            ### stack to find duplicates to figure out which coordinate matched!!!
            stack = np.vstack([expanded, coords_end_points])
            unq, count = np.unique(stack, axis=0, return_counts=True)
            coord_root = unq[count>1]
        
            #print(coord_root)
            
    
    """ next find segment tied to each branchpoint by searching the +/- 1 neighborhood for matching indices
    """
    all_neighborhoods, all_hood_first_last, root_neighborhood = get_neighborhoods(degrees, coord_root=coord_root)
    
    
    
    if len(root_neighborhood) == 0:
        return [], []
        
   
    """ Create tree """
    depth = 0                 
    tree_df, all_children = treeify(tree_df, depth, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = 0, parent= -1)                 

    return tree_df, all_children



""" 
   Gets the coordinates associated with the node at node_idx                 
"""
def get_next_coords(tree, node_idx, num_parents):

    parent_coords = get_parent_nodes(tree, start_ind=node_idx, num_parents=num_parents, parent_coords=[])
    
    if len(parent_coords) > 0:  # check if empty
        parent_coords = np.vstack(parent_coords)
    
    """ Get center of crop """
    cur_coords = []; 
    
    
    ### HACK: do this instead if neighborhood is just single coordinate
    if len(tree.start_be_coord[node_idx]) == 3:
        ### adding starting node
        cur_be_start = tree.start_be_coord[node_idx];     centroid_start = cur_be_start
        cur_coords.append(cur_be_start)
           
        ### Get middle of crop """
        coords = tree.coords[node_idx]
        cur_coords.append(coords)
        
        ### Get end of crop if it exists"        
        cur_be_end = tree.end_be_coord[node_idx];     centroid_end = cur_be_end
        cur_coords.append(cur_be_end)      
                    
        
    else:
    
        ### Get start of crop
        cur_be_start = tree.start_be_coord[node_idx]
          
        ### adding starting node
        centroid_start = cur_be_start[math.floor(len(cur_be_start)/2)]
        cur_coords.append(centroid_start)
           
        ### Get middle of crop """
        coords = tree.coords[node_idx]
        cur_coords.append(coords)
        
        ### Get end of crop if it exists
        if not np.isnan(tree.end_be_coord[node_idx]).any():   # if there's no end index for some reason, use starting one???
            """ OR ==> should use parent??? """             
            cur_be_end = tree.end_be_coord[node_idx]
            
            centroid_end = cur_be_end[math.floor(len(cur_be_end)/2)]
            cur_coords.append(centroid_end)      
                    
        else:
            ### otherwise, just leave ONLY the start index, and nothing else
            # cur_coords = centroid
            # cur_be_end = cur_be_start
            
            
            print('ERROR: NO END COORDINATE DETECTED')
            #zzz
      
    cur_coords = np.vstack(cur_coords)
    
    if np.shape(cur_coords)[1] == 1:
        cur_coords = np.transpose(cur_coords)
        
        
    return cur_coords, cur_be_start, cur_be_end, centroid_start, centroid_end, parent_coords
                  



""" expand coords into a neighborhood """
def expand_coord_to_neighborhood(coords, lower, upper):
    neighborhood_be = []
    for idx in coords:
        for x in range(-lower, upper):
            for y in range(-lower, upper):
                for z in range(-lower, upper):
                    new_idx = [idx[0] + x, idx[1] + y, idx[2] + z]
                    neighborhood_be.append(new_idx)    
                    
    return neighborhood_be


""" Get neighborhoods from an image ==> include scaling??? """
def get_neighborhoods(degrees, coord_root=0, scale=0, box_xyz=0, box_over=0, order=0, width=1000000000, height=100000000, depth=100000000):
      only_segments = np.copy(degrees); only_segments[only_segments != 2] = 0
      only_branch_ends = np.copy(degrees); only_branch_ends[only_branch_ends == 2] = 0; only_branch_ends[only_branch_ends > 0] = 3; 
      
      ### convert branch and endpoints into a list with +/- neihgbourhood values
      labels = measure.label(only_branch_ends)
      cc_be = measure.regionprops(labels)

      all_neighborhoods = []
      root_neighborhood = []
      
      for ind, branch_end in enumerate(cc_be):
          coords = branch_end['coords']
          
          neighborhood_be = expand_coord_to_neighborhood(coords, lower=1, upper=2)

          if (np.vstack(neighborhood_be)[:, None] == coord_root).all(-1).any():
              root_neighborhood.append(np.vstack(neighborhood_be))
              #print(ind)
              
              
          else:
              neighborhood_be = np.vstack(neighborhood_be)
              if scale:
                  
                  neighborhood_be = scale_coords_of_crop_to_full(neighborhood_be, box_xyz, box_over)
                  
              all_neighborhoods.append(neighborhood_be)
              
                      
      ### convert segments into just coords and ALSO get there neighborhoods for their FIRST and LAST indices
      labels = measure.label(only_segments)
      cc_segs = measure.regionprops(labels)
      
      all_hood_first_last = []          
      idx = 0
      for seg in cc_segs:
          coords = np.vstack(coords)
          coords = seg['coords']   
          
          """ ORDER THE LIST """
          #if order:
          #     coords = order_skeleton_list(coords)
          
          
          if scale:
              coords = scale_coords_of_crop_to_full(coords, box_xyz, box_over)
              
          all_hood_first_last.append(coords)
          idx += 1
      
      return all_neighborhoods, all_hood_first_last, root_neighborhood




""" Check size limits """
def check_limits(all_neighborhoods, width_tmp, height_tmp, depth_tmp):
    
        """ Make sure nothing exceeds size limits """
        idx = 0; 
        for neighbor_be in all_neighborhoods:
            
            if len(neighbor_be) > 0:
                if np.any(neighbor_be[:, 0] >= width_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 0] >= width_tmp), 0] = width_tmp - 1
                    
                if np.any(neighbor_be[:, 1] >= height_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 1] >= height_tmp), 1] = height_tmp - 1
    
                if np.any(neighbor_be[:, 2] >= depth_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 2] >= depth_tmp), 2] = depth_tmp - 1
            idx += 1
            
        return all_neighborhoods   
    



""" Create tree """
def treeify(tree_df, depth, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = 0, parent= -1, start=0, width_tmp=1000000000, height_tmp=100000000, depth_tmp=100000000):


        """ Make sure nothing exceeds size limits """
        all_neighborhoods = check_limits(all_neighborhoods, width_tmp, height_tmp, depth_tmp)
        all_hood_first_last = check_limits(all_hood_first_last, width_tmp, height_tmp, depth_tmp)


        # IF ROOT (depth == 0) ==> then use root neighborhood
        if len(tree_df) == 0 and not start:
            cur_be = np.vstack(root_neighborhood[0])
        elif start:
            cur_be = np.vstack(root_neighborhood)   ### start by setting the current branch endpoint to be the root
        else:
            idx_parent_df = np.where(tree_df.cur_idx == parent)
            cur_be = np.vstack(tree_df.end_be_coord[idx_parent_df[0][0]])

        """ find next segment of segmentation and compare all segments to cur_be (the current branch_endpoint
                
                if there 
        """
             
        all_children = [];
        for idx_cur_seg in range(len(all_hood_first_last)):
            if not np.asarray(all_hood_first_last[idx_cur_seg]).any():
                continue   # skip if empty
                
    
            cur_seg = np.vstack(all_hood_first_last[idx_cur_seg])
            
            """
                if there is a match: then add as new node to tree, where:
                        cur_be ==> becomes start_be_coord
                        
            """
            if (cur_seg[:, None] == cur_be).all(-1).any():
                
                num_missing = []
                if len(tree_df) > 0:
                    
                    ### First check if there is a missing value, because fill that first, otherwise, just do max + 1
                    lst = np.asarray(tree_df.cur_idx)
                    num_missing = [x for x in range(lst[0], lst[-1]+1) if x not in lst] 
                    if len(num_missing) > 0:
                        cur_idx = num_missing[0]
                    else:                    
                        cur_idx = np.max(tree_df.cur_idx[:]) + 1;  

                full_seg_coords = np.vstack(cur_seg)
                

               
                ### ADD NEW NODE TO TREE
                new_node = {'coords': full_seg_coords, 'parent': parent, 'child': [], 'depth': depth, 'cur_idx': cur_idx, 'start_be_coord': cur_be, 'end_be_coord': np.nan, 'visited': np.nan}
                
                ### if it's a deleted node, then add back into the location it was deleted from!!!
                if len(num_missing) > 0:
                    tree_df.loc[cur_idx] = new_node
                    tree_df = tree_df.sort_index()
                    ### else, add it to the end of the list
                else:
                    tree_df = tree_df.append(new_node, ignore_index=True)
                
                
                """ Finally, must find the "end_be" to complete the node of the tree
                """
                ### find next be
                next_be = []; all_neighborhoods_tmp = all_neighborhoods
                
                isempty = 1
                for idx_cur_be in range(len(all_neighborhoods)):
                    if not np.asarray(all_neighborhoods[idx_cur_be]).any():
                        continue # skip if empty
                    search_be = np.vstack(all_neighborhoods[idx_cur_be])
                    
                    if (cur_seg[:, None] == search_be).all(-1).any():        
                        next_be.append(search_be)
                        
                        # delete the neighborhood we currently assessed
                        all_neighborhoods_tmp[idx_cur_be] = [];
                        isempty = 0
                        
                #print(cur_idx)
                

                # delete the neighborhood we currently assessed
                all_hood_first_last[idx_cur_seg] = []                
                
                if not isempty and np.asarray(np.vstack(next_be)).any():
                    # append children to send to previous call
                    all_children.append(cur_idx)
                    
                    
                    next_be = np.vstack(next_be)
                    idx_parent_df = np.where(tree_df.cur_idx == cur_idx)
                    tree_df.end_be_coord[idx_parent_df[0][0]] = next_be
                    
                    
                    """ organize coordinates of full segmentation here??? """
                    to_sort = np.vstack((cur_be[int(len(cur_be)/2)], full_seg_coords))
                    to_sort = np.vstack((to_sort, next_be[int(len(next_be)/2)]))
                    
              
                    if len(to_sort) > 3:
                        ordered = order_coords(to_sort, idx_start=0)   
                        
                        tree_df.coords[idx_parent_df[0][0]] = ordered
                    else:
                        print('skipped')
                                                
                    # recurse
                    tree_df, next_children = treeify(tree_df, depth + 1, 
                                          root_neighborhood, all_neighborhoods_tmp, all_hood_first_last, cur_idx=cur_idx, parent=cur_idx,
                                          width_tmp=width_tmp, height_tmp=height_tmp, depth_tmp=depth_tmp) 
    
                    # add all children from next call
                    tree_df.child[cur_idx].append(next_children)
                    
                    
                    """ OTHERWISE, delete the node because there is no matching end point
                            ***b/c don't allow cyclilization, only linear trees
                    
                    """
                else:
                    #tree_df = tree_df.drop(cur_idx,inplace=False)
                    #tree_df = tree_df.reset_index(drop=True)
                    
                    
                    tree_df.visited[cur_idx] = -1
                    
                    
                    # tree_df.coords[cur_idx] = []
                    # tree_df.start_be_coord[cur_idx] = []
                    # tree_df.end_be_coord[cur_idx] = []
                    # tree_df.parent[cur_idx] = []
                    # tree_df.child[cur_idx] = []
                    
                    
                    print('dropped')
                    


        ### convert empty lists to "nan" ONLY in the end_be_coord column
        # if parent == -1 or start == 1:
        #     #tree_df = tree_df.mask(tree_df.applymap(str).eq('[]'))
        #     #tree_df = tree_df.mask(tree_df.applymap(str).eq('[[]]'))
        #     a = 1

        return tree_df, all_children
    
import math


            


    
    
""" Plot tree """
def show_tree(tree_df, im):
    width_tmp = np.shape(im)[0]
    height_tmp = np.shape(im)[1]
    depth_tmp = np.shape(im)[2]

    all_segments = np.asarray(tree_df.coords[:])
    indices = np.asarray(tree_df.cur_idx)
    start_be_coord = np.asarray(tree_df.start_be_coord[:])

    end_be_coord = np.asarray(tree_df.end_be_coord[:])

    for seg, ind, start_be, end_be in zip(all_segments, indices, start_be_coord, end_be_coord):
        
        im[seg[:, 0], seg[:, 1], seg[:, 2]] = ind + 1 # because dont want non-zero
        
        
        ### if only single coordinate, then don't check
        if len(start_be) == 3:
            im[start_be[0], start_be[1], start_be[2]] = 1           
        else:
            start_be = check_limits([start_be], width_tmp, height_tmp, depth_tmp)[0]  ### check to make sure none of the dilated neighborhoods go out of bounds
            im[start_be[:, 0], start_be[:, 1], start_be[:, 2]] = 1

        
        if not np.isnan(end_be).any() and np.asarray(end_be).any():
            ### if only single coordinate, then don't check
            if len(start_be) == 3:
                im[end_be[0], end_be[1], end_be[2]] = 1           
            else:       
                end_be = check_limits([end_be], width_tmp, height_tmp, depth_tmp)[0]
                im[end_be[:, 0], end_be[:, 1], end_be[:, 2]] = 2
    return im


""" Get ALL coords of tree """
def show_tree_FAST(tree_df):
    all_segments = np.asarray(tree_df.coords[:])
    indices = np.asarray(tree_df.cur_idx)
    #start_be_coord = np.asarray(tree_df.start_be_coord[:])
    all_segs = np.vstack(all_segments)
    #im[all_segs[:, 0], all_segs[:, 1], all_segs[:, 2]] = 1 # because dont want non-zero
        
    return all_segs


""" Get all coords of tree EXCEPT for... """
def show_tree_FAST_drop_index(tree_df, drop_id):
    
    copy_tree = tree_df.copy()
    copy_tree = copy_tree.drop([drop_id])
    all_segments = np.asarray(copy_tree.coords[:])
    indices = np.asarray(copy_tree.cur_idx)
    #start_be_coord = np.asarray(tree_df.start_be_coord[:])
    all_segs = np.vstack(all_segments)
    #im[all_segs[:, 0], all_segs[:, 1], all_segs[:, 2]] = 1 # because dont want non-zero
        
    return all_segs