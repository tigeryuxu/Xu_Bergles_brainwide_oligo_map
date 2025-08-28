#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:34:50 2020

@author: user
"""

""" Convert imageJ file to .swc with correct radii thickness 

        read in file
        output file
"""
import glob, os
import numpy as np
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order



""" Load all trees file """
import csv
def load_all_trees(tree_csv_path):
    all_trees = []
    with open(tree_csv_path + 'all_trees.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row_num, row in enumerate(spamreader):
            print(', '.join(row))
            
            
            if row_num % 2 == 0:
                parents = []
                size = []
                for num, entry in enumerate(row):
                    if num == 0:
                        #im_name = '.tif'.join(entry.split('.tif')[0:-1])
                        im_name = entry.split('.tif')[0]
       
                        
                    elif num == 1:
                        continue
                    elif num == 2 or num == 3 or num == 4:
                        size.append(int(entry))
                    
                    else:
                        if not entry == '':
                            parents.append(int(entry))

            else:
                orig_idx = []
                for num, entry in enumerate(row):
                    if num == 0 or num == 1 or num == 2 or num == 3 or num == 4:
                        continue
                                        
                    else:
                        if not entry == '':
                            orig_idx.append(int(entry))    
                            
                tree_entry = dict(im_name = im_name, orig_idx = np.transpose(orig_idx), parents = np.transpose(parents), size=size)
                all_trees.append(tree_entry)  
                
    return all_trees



""" Save tree as .swc output file:
    
    
    columns:
        index | type (soma, dendrite, ect...) | X | Y | Z |radius (0.5 or 0) | parent   (-1) for root
    
    Types:
        0 - undefined
        1 - soma
        2 - axon
        3 - (basal) dendrite
        4 - apical dendrite
        5+ - custom
    
    ***IN microns!!!
    
    """        
def save_tree_to_swc(tree, s_path, filename='output.swc', scale_xy=0.20756792660398113, scale_z=1):
    all_vertices = []
    for index, vertex in tree.iterrows():
         if not np.isnan(vertex.start_be_coord).any():
              all_vertices.append(vertex.start_be_coord)        

    all_vertices = np.vstack(all_vertices)  # stack into single array
    
    all_x = all_vertices[:, 0]
    all_y = all_vertices[:, 1]
    all_z = all_vertices[:, 2]

    col1_index = np.asarray(tree.cur_idx)
    col2_type = [2] * len(col1_index)
    col3_x = all_y * scale_xy           ### INVERT X and Y!!!
    col4_y = all_x * scale_xy
    col5_z = all_z * scale_z
    col6_radi = [0.5] * len(col3_x)
    col7_parent = np.asarray(tree.parent)

    full_arr = np.transpose(np.array([col1_index, col2_type, col3_x, col4_y, col5_z, col6_radi, col7_parent]))
    
    
    """ If need to set a soma point """
    full_arr[0][1] = 1
    
    
    
    datafile_path = s_path + filename
    with open(datafile_path, 'w+') as datafile_id:
        np.savetxt(datafile_id, full_arr, fmt=['%d','%d', '%.5f','%.5f', '%.5f','%.5f', '%d'])





""" Turn tree into .obj file

     v x y z
     l a b c d e f...
     

"""

def linearize_tree(tree, cur_idx, list_lines):
       if len(tree.child[cur_idx]) == 0:
            #print(len(tree.child[cur_idx]))
            return list_lines  # hit bottom of tree
       
       else:
            child_idx = 0;
            
            for child in tree.child[cur_idx]:
                 #print(child_idx)
                 #print(child)
                  
                 if child_idx == 0:   ### append directly if it's the first child
                      list_lines[-1].append(child + 1)
                      list_lines = linearize_tree(tree, child, list_lines)
                 else:
                      list_lines.append([cur_idx + 1])   ### include parent in line to be complete (at least 2 points to form each line)
                      list_lines[-1].append(child + 1)
                      
                      list_lines = linearize_tree(tree, child, list_lines)
                      
                 
                 child_idx += 1;
  
       return list_lines  


def save_tree_to_obj(all_trees, s_path, filename, get_mid=1):
    ### (0) actually have to combine all branches together first
    all_starting_indices = [];
    idx = 0;
    for tree in all_trees:
    
        """ first clean up parent/child associations """
        for index, vertex in tree.iterrows():
             cur_idx = vertex.cur_idx
             children = np.where(tree.parent == cur_idx)
             
             vertex.child = children[0]
                              
        if idx == 0:
            all_trees_appended = all_trees[0]
            all_starting_indices.append(0)
            idx += 1
            continue
             
        for r_id, row in enumerate(tree.child):                
                tree.child[r_id] = np.add(tree.child[r_id], len(all_trees_appended)).tolist() 
                
        #tree.child = tree.child + len(all_trees_appended) 
        tree.parent = tree.parent + len(all_trees_appended) 
        tree.cur_idx = tree.cur_idx + len(all_trees_appended) 
        
        all_trees_appended = all_trees_appended.append(tree, ignore_index=True)
        
        all_starting_indices.append(len(all_trees_appended))
        
        idx += 1
    
    
    file = open(s_path + filename, "w")
    ### (1) first get vertices, which is just end coords + root coord
    
    ### ***must also include the root index??? but then would misalign with the rest of the line coords???
         ###***so... maybe add as -1, or just don't add it???
              ###***will result in extra error? or just remove from MATLAB output as well (from simple neurite tracer)
    
    ### (a) must first actually make sure that all children are associated with their parents
    all_vertices = []
    for index, vertex in all_trees_appended.iterrows():
         # cur_idx = vertex.cur_idx
         # children = np.where(all_trees_appended.parent == cur_idx)
         
         # vertex.child = children[0]
          
         if not np.isnan(vertex.start_be_coord).any():
              
             if get_mid == 1:
                 all_vertices.append(vertex.start_be_coord)
    
             else:
                 all_vertices.append(vertex.start_be_coord)
    
    ### (3) save as .obj file
    for v in all_vertices:
      file.write("v %d %d %d\n" %(v[0], v[1],  v[2]))                
    
    
    ### (2) then, find order with which coords are connected
         ### ***must be + 1 to child b/c 0 vertex doesn't exist
    if not get_mid:
        all_starting_indices = [1]
                 
    for cur_idx in all_starting_indices:
        
        if cur_idx == all_starting_indices[-1]:   # skip last one
            continue
        
        list_lines = [[cur_idx + 1]];
        list_lines = linearize_tree(all_trees_appended, cur_idx, list_lines)
      
        #file.write("World\n")
        #for l in list_lines:
        file.write("l ")
        file.write("\nl ".join(" ".join(map(str, x)) for x in list_lines))
        file.write("\n")
        print()
         
    file.close()




""" Fixes the radius of SNT outputs """
import pandas as pd
def fix_SNT_outputs_radii():
    swc_path = '/media/user/storage/Data/(1) snake seg project/Traces files/swc files/'
    swc_files = glob.glob(os.path.join(swc_path,'*.swc'))
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    swc_files.sort(key = natsort_key1)
    
    
    scale_xy=0.20756792660398113
     
    for filename in swc_files:           
        save_name = filename.split('.')[0]
        save_name = save_name + '_reformated.swc'
    
        save_file = open(save_name, "w+")
        
        first_line = 0
        all_lines = []
        with open(filename) as fp:
            for line in fp:
                
                line = line.split()  ### splits into individual values
                if line[0] == '#':   ### skip if it's a comment
                    continue
                

                """ If reformat all into axons """                
                line[1] = 2   ### make all into axons
                
                """ NEEDS A SOMA FOR MORPHOPY TO WORK"""
                if first_line == 0:
                    line[1] = 1
                    first_line += 1
                    
                all_lines.append(line)
                
                
                """ TIGER NEWLY ADDED: SEEMS LIKE Z-axiz wrongly scaled???"""
                line[4] = str(float(line[4]) / scale_xy)
                
        
                line[-2] = 0.5   ### replace RADII value
                save_file.write("%s %s %s %s %s %.2f %s\n" %(line[0], line[1],  line[2], line[3],  line[4],line[5],  line[6]))      
                             
        save_file.close()   
        
        """ save as .obj file """
        ### save the file into a tree as well:
        columns = {'coords', 'parent', 'child', 'depth', 'start_be_coord', 'end_be_coord', 'cur_idx', 'visited'}
        tree_df = pd.DataFrame(columns=columns)
                    
        all_lines = np.vstack(all_lines)
        all_lines = all_lines.astype(np.float)
        
        """ add to dataframe """
        for counter, row in enumerate(all_lines):
            new_node = {'coords': [], 'parent': float(row[6]), 'child': [], 'depth': [], 'cur_idx': float(row[0]), 'start_be_coord': [float(row[2]) * (1/scale_xy), float(row[3]) * (1/scale_xy), float(row[4])], 'end_be_coord': [], 'visited': np.nan}
            tree_df = tree_df.append(new_node, ignore_index=True)
            
            ### also need to find all of it's children
            
            idx_children = np.where(all_lines[:, 6] == tree_df.cur_idx[counter])[0]
            
            tree_df.child[counter] = idx_children
    
            
        #s_path = swc_path
        save_tree_to_obj([tree_df], s_path = '', filename=filename.split('.')[0] + '_object.obj', get_mid=0)           
        
      
        
        
        
        
        
        