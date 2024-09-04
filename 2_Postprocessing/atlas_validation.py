# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:46:29 2020

@author: tiger
"""

from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Neuroimmunology Unit
"""

# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL numexpr!!!
 
@author: Tiger


"""

import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import pickle as pickle
import os
import scipy
from natsort import natsort_keygen, ns

#from plot_functions_CLEANED import *
#from data_functions_CLEANED import *
#from data_functions_3D import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import tkinter
from tkinter import filedialog
import os
    
import tifffile as tiff

                          
#import nibabel as nib
import json

import pandas as pd
import random as random            

### Makes it so that svg exports text as editable text!
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'    



truth = 0

def plot_max(im, ax=0):
     max_im = np.amax(im, axis=ax)
     plt.figure(); plt.imshow(max_im[:, :])
     return max_im
     





input_path = '/media/user/8TB_HDD/Validate_atlas/'






""" Loop through all the folders and do the analysis!!!"""
# for input_path in list_folder:
foldername = input_path.split('/')[-2]
sav_dir = input_path + '/' + foldername + '_plot_validation'
 
""" Load filenames from tiff """
images = glob.glob(os.path.join(input_path,'*_selected_labels.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
examples = [dict(input=i,truth=i.replace('_input_im_CROP.tif','_seg_CROP.tif'), 
                 maskrcnn=i.replace('_input_im_CROP.tif','_cleaned_NEW_CROP.tif'),
                 maskrcnn_8bit=i.replace('_input_im_CROP.tif','_cleaned_NEW_RGB_to_8bit_CROP.tif'),
                 ) for i in images]


try:
    # Create target Directory
    os.mkdir(sav_dir)
    print("Directory " , sav_dir ,  " Created ") 
except FileExistsError:
    print("Directory " , sav_dir ,  " already exists")
    
sav_dir = sav_dir + '/'


atlas_labels = tiff.imread(input_path + 'Atlas_labels.tif')


all_dict = []
for file in examples:
    input_name = file['input']
    input_im = tiff.imread(input_name)
    
    filename = input_name.split('/')[-1].split('.')[0]
    
    dict_df = {'RSPv':0, 'CC':0, 'RSPd1':0, 'SLM':0, 'Hilus':0, 'IB1':0,
               'IB2':0, 'SSp1':0, 'HY1':0, 'HY2':0, 'MB':0, 'PAA':0} #'exp_name':filename[:4]}
    
    for label_num in range(1, np.max(atlas_labels) + 1):
        
        scale = 20  # 20um/pixel
        
        # print(label_num)
        atlas_coord = np.transpose(np.where(atlas_labels == label_num)) * scale
        val_coord = np.transpose(np.where(input_im == label_num)) * scale
        
        dist = np.linalg.norm(atlas_coord - val_coord)
        
        dict_df[list(dict_df.keys())[label_num - 1]] = dist
        
        
    all_dict.append(dict_df)


df = pd.DataFrame(all_dict)
    

df = pd.melt(df.transpose().reset_index(), id_vars=['index'])


df = df.drop(np.where(df['index'] == 'RSPv')[0]).reset_index()
df = df.drop(np.where(df['index'] == 'Hilus')[0])

import seaborn as sns
palette = 'Set2'
fontsize=14
order=['CC', 'RSPd1','SSp1', 'SLM', 'IB1', 'IB2', 'SSp1', 'HY1', 'HY2', 'MB', 'PAA']

plt.figure(figsize=(4,3))
sns.pointplot(df, x='index', y='value', c='k',
      errorbar='se', capsize=0.2, linestyle="none",
      # marker="_", 
      markersize=4, order=order)
ax = sns.stripplot(df, x='index', y = 'value', jitter=True, palette=palette, order=order)

# sns.move_legend(ax, loc='upper left', frameon=False, title='', fontsize=8)
# plt.xlim([-xlim, xlim])

plt.ylim([0, 100])
plt.yticks(fontsize=fontsize - 2)
# plt.yticks([])
plt.xticks(fontsize=fontsize - 2)

plt.xticks(rotation=45) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('', fontsize=fontsize)
plt.ylabel('Deviation from atlas ($\mu$m)', fontsize=fontsize)
plt.tight_layout()


plt.savefig(sav_dir + '_compare_fiducial_landmarks.png', format='png', dpi=300)
plt.savefig(sav_dir + '_compare_fiducial_landmarks.svg', format='svg', dpi=300)

    






















