#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:03:32 2023

@author: user
"""
import z5py

import glob, os
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
import pandas as pd
# pd.options.mode.chained_assignment = None  ### disable warning

import tifffile as tiff

import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean

from skimage.measure import label, regionprops, regionprops_table

import json
import matplotlib.pyplot as plt
    
from postprocess_func import *
import seaborn as sns
from scipy import stats   

import sys
sys.path.append("..")

from get_brain_metadata import *



import matplotlib.pyplot as plt

### Makes it so that svg exports text as editable text!
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'    


#%% Also load in control brain
# list_brains = get_metadata(mouse_num = ['M296'])
# input_path = list_brains[0]['path']
# name_str = list_brains[0]['name']
# n5_file = input_path + name_str + '.n5'
# downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
# atlas_dir_c1 = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
# f = z5py.File(n5_file, "r")
# dset_c1 = f['setup0/timepoint0/s0']  




#%% 5xFAD and control data
list_brains = get_metadata(mouse_num = ['M243', 'M244', 'M234', 'M235', 'M217', ### 5xFAD
                                        'M304', 'M242', 'M297', 'M296'])        ### control

# Use below if you want to make the cuprizone graph for comparisons
# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267', ### cuprizone
#                                         'M254', 'M256', 'M260',
#                                         'M299', 'M223', #'M229',
#                                         'M126', 'M127'
#                                         ])        ### control




cloudreg = 0

ANTS = 1

#%% Parse the json file so we can choose what we want to extract or mask out
reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'


ref_atlas = tiff.imread(reference_atlas)
ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)


ref_atlas_ids = np.unique(ref_atlas)

cc_allen = regionprops(ref_atlas, cache=False)
cc_labs_allen = [region['label'] for region in cc_allen]
cc_labs_allen = np.asarray(cc_labs_allen)


with open('../atlas_ids/atlas_ids.json') as json_file:
    data = json.load(json_file)
 
     
data = data['msg'][0]

#%% Also extract isotropic volumes from reference atlas DIRECTLY - to avoid weird expansion factors
print('Extracting main key volumes')
keys_dict = get_ids_all(data, all_keys=[], keywords=[''])  
main_keys = pd.DataFrame.from_dict(keys_dict)
main_keys = get_atlas_isotropic_vols(main_keys, ref_atlas, atlas_side='_W', XY_res=20, Z_res=20)
main_keys['atlas_vol_R'] = main_keys['atlas_vol_W']/2
main_keys['atlas_vol_L'] = main_keys['atlas_vol_W']/2





#%% Loop through each experiment
all_keys_dfs = []
fimbria_all = []
for exp in list_brains:
    keys_df = main_keys.copy(deep=True)

    keys_df['exp'] = exp['exp']
    keys_df['mouse_num'] = exp['num']



    input_path = exp['path']
    name_str = exp['name']
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
    
    
    f = z5py.File(n5_file, "r")
    dset = f['setup0/timepoint0/s0']
    
    # dset_halo = f['setup1/timepoint0/s0']

    # dset_congo = f['setup2/timepoint0/s0']    



    ### NEW WITH ALLEN
    if exp['thresh'] != 0: ### for DIVIDED BY MYELIN
    
        ### WHOLE BRAIN registered to Allen --- use this for cerebellum!!!
        # atlas_dir = downsampled_dir + name_str + 'allen_mouse_20um_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'
        atlas_WHOLE_dir = downsampled_dir + name_str + 'allen_mouse_20um_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default_PADDED_50/'
    
        ### This is just cortex
        if ANTS:
            ### cortex registered to Allen
            # atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str + 'allen_mouse_CORTEX_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
       
            ## cortex registered using MYELIN brain average template
            atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'

            # ### cortex registered using OUR OWN CUBIC AUTOFLUORESCENCE
            # atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_20um_CUBIC_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'


    else:
        
        ### NEED A ATLAS_WHOLE_DIR for CUBIC brain!!! i.e. whole CUBIC template brain, including cerebellum
        # atlas_dir = downsampled_dir + name_str + 'allen_mouse_20um_CUBIC_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'
        atlas_WHOLE_dir = downsampled_dir + name_str + 'allen_mouse_20um_CUBIC_FULLBRAIN_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default_PADDED_50/'
           
        
        ### This is just cortex
        if ANTS:
            atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_20um_CUBIC_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
            print('CUBIC reference')



    scaled = 1

    allen_dir = downsampled_dir + name_str + '_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
    analysis_dir = input_path + name_str + '_postprocess_CONGO/'
    

    myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_n4_down1_resolution_20_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
    auto_path = glob.glob(os.path.join(downsampled_dir, '*_ch1_n4_down1_resolution_20_PAD.tif'))[0]
    
 
    
 
    reg_whole_brain = os.path.join(atlas_WHOLE_dir + 'downsampled_standard_' + name_str + '_level_s3_ch0_n4_down1_resolution_20_PAD.tiff') 
 
 
    reg_myelin_path = os.path.join(atlas_dir + 'downsampled_standard.tiff')
 
    pad = True
    
    XY_res = 1.152035240378141
    Z_res = 5
    
    
    res_diff = XY_res/Z_res
    

    
    """ Loop through all the folders and do the analysis!!!"""
    #filename = n5_file.split('/')[-2]
    filename = n5_file.split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename) 
    sav_dir = input_path + '/' + filename + '_postprocess_CONGO'
    
    sav_dir_regular = input_path + '/' + filename + '_postprocess' #updated the folder name to include the new threshold
    
    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(analysis_dir,'*_df.pkl'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i.replace('_df.pkl','_input_im.tif'),pkl=i, segmentation=i.replace('_df.pkl','_segmentation_overlap3.tif'),
                     shifted=i.replace('_df.pkl', '_shifted.tif')) for i in images]
     
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("\nSave directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    

    #%% Load
    
    #autofluor = tiff.memmap(auto_path)
    # myelin = tiff.imread(myelin_path)
    myelin = tiff.imread(reg_whole_brain)


    
    
    #%% Load registered atlas
    print('Loading registered atlas')
    #dirlist = os.listdir(input_path)
    #atlas_dir = [s for s in dirlist if 'allen_mouse_25um' in s][0]
    
    
    # Get all region IDs present in the registered atlas
     
    region_means = []
    region_sums = []

    myelin_thresh = myelin.astype(np.float32)
    myelin_thresh[myelin_thresh < 100] = np.nan

    
    from tqdm import tqdm
    print(f'Processing myelin signal for: {name_str}')    
    for region_id in tqdm(ref_atlas_ids, desc='Extracting myelin stats'):
        if region_id == 0:
            continue  # skip background
    
        mask = ref_atlas == region_id
        masked_vals = myelin_thresh[mask]
    
        if masked_vals.size == 0 or np.all(np.isnan(masked_vals)):
            mean_val = np.nan
            sum_val = np.nan
        else:
            mean_val = np.nanmean(masked_vals)
            sum_val = np.nansum(masked_vals)
    
        region_means.append((region_id, mean_val))
        region_sums.append((region_id, sum_val))
            
    # Convert to DataFrames
    mean_df = pd.DataFrame(region_means, columns=['region_id', 'mean_myelin'])
    sum_df = pd.DataFrame(region_sums, columns=['region_id', 'sum_myelin'])
    # Merge with keys_df
    keys_df = keys_df.merge(mean_df, left_on='ids', right_on='region_id', how='left')
    keys_df = keys_df.merge(sum_df, on='region_id', how='left')
    keys_df.drop(columns='region_id', inplace=True)


    ### Weight myelin for accumulation in next step
    keys_df['weighted_myelin'] = keys_df['atlas_vol_W'] * keys_df['mean_myelin']    






    drop_ids = []
    #%%% REMOVE ADDITIONAL TORN TISSUE REGIONS - as specified in metadata (including replacements if given hemisphere)
    # keys_tmp = keys_df.copy()
    # keys_df['acronym'] = keys_tmp['acronym']
    keys_df['dataset'] = exp['num']
    keys_df['exp'] = exp['exp']
    keys_df['sex'] = exp['sex']
    keys_df['age'] = exp['age']   
       
    if 'exclude' in exp:
        print('removing torn regions')
        # zzz
        
        exclude = exp['exclude']
        

        ### ADDITIONAL AREAS TO EXCLUDE FOR ALL BRAINS        
        additional_exclude = [
            ['ME', 'B'],   
            
            ['RCH', 'B'],   ### Hypothalamaic
            ['VMH', 'B'],
            ['Xi', 'B'],    ### Thalamus
            
            ['SCO', 'B'],   ### Midbrain
            ['LING', 'B'],  ### Cerebellum
            
            ]
        
        exclude = exclude + additional_exclude
        
        
        ### do the entire brain
        if exclude[0][0] == 'all':
                # keys_df['num_OLs_L'] = keys_df['num_OLs_' + exclude[0][1]]
                # keys_df['num_OLs_R'] = keys_df['num_OLs_' + exclude[0][1]]
                # keys_df['num_OLs_W'] = keys_df['num_OLs_' + exclude[0][1]] * 2  
                print('Skipping right now')
        
        ### Do by region
        else:
            all_ids = []
            for region in exclude:
                
                
                reg_ids = get_sub_regions_by_acronym(keys_df, child_id=[], sub_keys=[], reg_name=region[0])
                
                ### if we decide to only numbers from one hemisphere, then overwrite data of other hemisphere
                if region[1] != 'B' and region[1] != '':
                    
                    # keys_df.loc[reg_ids, 'num_OLs_L'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]]
                    # keys_df.loc[reg_ids, 'num_OLs_R'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]]
                    # keys_df.loc[reg_ids, 'num_OLs_W'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]] * 2
                    print('Also not deleting right now if want a hemisphere')
        
                else:
                
                    drop_ids = np.concatenate((drop_ids, reg_ids))
                

    #%%%% REMOVE HINDBRAIN ENTIRELY
    
    hindbrain_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Hindbrain')
    drop_ids = np.concatenate((drop_ids, hindbrain_ids))
        
    
    #% ALSO REMOVE Midbrain, behavioral state related (deep structures)
    
    mid_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Midbrain, behavioral state related')
    drop_ids = np.concatenate((drop_ids, mid_ids))
        
     
    ### Actually drop here
    children_to_erase = keys_df.iloc[drop_ids]['ids'].values
    
    
    
    
    keys_df = keys_df.drop(drop_ids)#.reset_index()

    new_childs = []
    for id_it, row in keys_df.iterrows():
        
        common = set(children_to_erase) & set(row['children'])
        a = [i for i in row['children'] if i not in common]
    
        new_childs.append(a)
    
    keys_df = keys_df.drop(columns=['children'])
    keys_df['children'] = new_childs  

    ### drop the redundant index columns
    # keys_df = keys_df.drop(columns=['level_0', 'index'])
    




    #%%%% sort by depth level and pool all values from children
    df_level = keys_df.sort_values(by=['st_level'], ascending=False, ignore_index=False)
    
    for i, row in df_level.iterrows():
        
        childs = row['children']
        
        if len(childs) > 0 and np.isnan(row['atlas_vol_W']):  ### if current row is NAN, then want to pool children, otherwise no
                    
            df_childs = pd.DataFrame()
            for child in childs:
                
                if len(np.where(df_level['ids'] == child)[0]) == 0:  ### means layer 6 was already deleted
                    continue
                id_c = np.where(df_level['ids'] == child)[0][0]
                
                child_row = df_level.iloc[id_c]
                
                child_df = pd.DataFrame([child_row])
                
                df_childs = pd.concat([df_childs, child_df], axis=0)
                
            df_sum = df_childs.sum(axis=0, numeric_only=True)
            
            df_sum = df_sum.drop(['ids', 'parent', 'st_level'])
            
            row[df_sum.index] = df_sum
            
            
            df_level.loc[i] = row  ### add row back in
          
    keys_df = df_level
    
    keys_df['mean_myelin'] = keys_df['weighted_myelin']/keys_df['atlas_vol_W']
    


    all_keys_dfs.append(keys_df)


    #%% Also extract and keep track of the fimbria for each dataset
    # sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='fimbria')
    # fimbria_ids = keys_df.iloc[sub_idx]['ids'].values
    
    # fimbria_mask = np.isin(ref_atlas, fimbria_ids)
    
    
    ### FIMBRIA MASK ALREADY CLEANED AND GENERATED
    fimbria_mask = tiff.imread('/home/user/.brainglobe/fimbria_masks_cleaned/fimbria_mask_ref_atlas_20um.tif')

    

    fimbria = np.copy(myelin)
    fimbria[fimbria_mask == 0] = 0
    
    fimbria_all.append(fimbria)
    
            
    # import napari
    # viewer = napari.Viewer()
    # # viewer.add_image(myelin)
    # viewer.add_image(fimbria_mask)
    # viewer.show(block=True)
    
    



    
# import napari
# viewer = napari.Viewer()
# viewer.add_image(myelin, colormap='gray', contrast_limits=[0, 2000])
#                   #name=list_brains[id_f]['num'])
# for id_f, fimbria in enumerate(fimbria_all):
    
#     # plot_max(fimbria)
#     plt.title(list_brains[id_f]['num'])
#     viewer.add_image(fimbria, colormap='turbo', contrast_limits=[350, 2000],
#                       name=list_brains[id_f]['num'])

# viewer.show(block=True)




import matplotlib.pyplot as plt
import seaborn as sns

# Style config
fontsize = 12
ticksize = 10
figsize = (6, 3)
palette = sns.color_palette("Set2")



exp_name = '5xFAD'


#%% Merge and plot --- STILL NEED TO REMOVE HINDBRAIN AND EXTRA GARBAGE
df_merged = pd.concat(all_keys_dfs)

### Declare plotting variables
palette = sns.color_palette("Set2")
### CAREFUL ---> df_means right now is pooled from ALL groups... for sorting...



from PLOT_FUNCTIONS_postprocess_compare import *
from collections import defaultdict

from scipy.ndimage import gaussian_filter
from fimbria_functions import * 
sav_fold = '/media/user/4TB_SSD/Plot_outputs_HALO/'

sav_fold_congo = '/media/user/4TB_SSD/Plot_outputs_CONGO/'

fontsize = 12


zzz



#%% Cuprizone volcano plot
if 'Cuprizone' in df_merged['exp'].values: 
    print('yes')
    exclude_nums = ['M267']
    df_filtered = df_merged[~df_merged['mouse_num'].isin(exclude_nums)]
    
    
    
    df_pval = volcano_compare(df_filtered, keys_df, compareby='exp', group1='P60', group2='Cuprizone',
                    compare_col = 'mean_myelin',
                    xplot='log2fold', thresh_pval=0.05, thresh_log2=0.3,
                    fontsize=fontsize, xlim=1, ylim=6, figsize=(3.2, 3))
    
    estimate_log2fc_for_target_d(df_pval, target_d=1.0)
    
    
    
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE_CUPRIZONE_myelin.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE_CUPRIZONE_myelin.svg', format='svg', dpi=300)
    





#%%% Volcano plot
exclude_nums = ['M217']
df_filtered = df_merged[~df_merged['mouse_num'].isin(exclude_nums)]



df_pval = volcano_compare(df_filtered, keys_df, compareby='exp', group1='6mos', group2='5xFAD',
                compare_col = 'mean_myelin',
                xplot='log2fold', thresh_pval=0.05, thresh_log2=0.3,
                fontsize=fontsize, xlim=1, ylim=6, figsize=(3.2, 3))

estimate_log2fc_for_target_d(df_pval, target_d=1.0)



plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE_5xFAD_myelin.png', format='png', dpi=300)
plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE_5xFAD_myelin.svg', format='svg', dpi=300)




#%% Density plots
# Cortex plot
names_to_plot = [
                  'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                 'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                 'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                 'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                 ]
palette = sns.color_palette("Set2", len(names_to_plot))

    
plot_global(df_filtered, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_5xFAD',
            dname='mean_myelin',
            ylim=450, figsize=(6, 3), fontsize=14)

    
    


#%% Find relationship between plaque load and myelin loss
def linear_regress(plot_vals_raw, sav_fold, exp_name, x='density_W', y='congo_density', figsize=(3.5, 3), xlim=None, ylim=None, xlabel=None, ylabel=None, fontsize=12, annotate_all=True, highlight_acronyms=None, annotate_top_n=None):
    
    # Filter out rows where plaque density is zero
    plot_vals = plot_vals_raw[plot_vals_raw[y] != 0].copy()


    plt.figure(figsize=figsize)
    marker_size = 20 if len(plot_vals) <= 500 else 8
    fig = sns.regplot(data=plot_vals, x=x, y=y, scatter_kws={'s': marker_size}, line_kws={'color': '#00A676'}, color='gray')

    # calculate slope, intercept, correlation
    slope, intercept, r_val, p_val, sterr = stats.linregress(
        x=fig.get_lines()[0].get_xdata(),
        y=fig.get_lines()[0].get_ydata()
    )

    r, p = stats.pearsonr(plot_vals.dropna()[x], plot_vals.dropna()[y])
    print(f"slope: {slope:.4f}, r: {r:.4f}, p: {p:.4e}")

    all_texts = []
    if annotate_all or highlight_acronyms is not None or annotate_top_n is not None:
        if annotate_top_n:
            top_rows = plot_vals.nlargest(annotate_top_n, y)
        for line in range(plot_vals.shape[0]):
            acronym = plot_vals['acronym'].iloc[line]
            if annotate_all or \
               (highlight_acronyms and acronym in highlight_acronyms) or \
               (annotate_top_n and plot_vals.iloc[line]['acronym'] in top_rows['acronym'].values):
                all_texts.append(plt.text(
                    plot_vals[x].iloc[line] + 0.01,
                    plot_vals[y].iloc[line],
                    acronym,
                    ha='center', va='center', size=10, color='black'))

    ax = plt.gca()
    ax.legend().set_visible(False)
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if xlim:
        plt.xlim(xlim)
    plt.xlabel(xlabel if xlabel else x, fontsize=fontsize)
    plt.ylabel(ylabel if ylabel else y, fontsize=fontsize)
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()

    if annotate_all or highlight_acronyms is not None or annotate_top_n is not None:
        adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='red'))

    plt.savefig(sav_fold + exp_name + '_plaque_vs_OL_density.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_plaque_vs_OL_density.svg', format='svg', dpi=300)



### also load in congo plaque counts
import pickle

# Save to pickle
with open(sav_fold_congo + 'df_congo.pkl', 'rb') as f:
    df_congo = pickle.load(f)



congo_df = df_congo
congo_df = congo_df[congo_df['exp'].isin(['5xFAD'])]

    
AD_df = df_filtered

AD_df = AD_df[AD_df['exp'].isin(['5xFAD', '6mos'])]


plot_vals = AD_df[['exp', 'acronym', 'mean_myelin', 'st_level']]

plot_vals = plot_vals.groupby(['exp', 'acronym']).mean().reset_index()

# Assuming df is your original DataFrame
group1 = plot_vals[plot_vals['exp'] == '6mos']
group2 = plot_vals[plot_vals['exp'] == '5xFAD']
    
# Merge group1 and group2 on the 'B' column (region)
merged = pd.merge(group1, group2, on='acronym', suffixes=('_6mos', '_5xFAD'))

# Then add in the 5xFAD
plot_vals = congo_df[['exp', 'acronym', 'density_W', 'st_level']]
plot_vals = plot_vals.groupby(['exp', 'acronym']).mean().reset_index()   

merged = pd.merge(merged, plot_vals, on='acronym', suffixes=('', '_congo'))

# Perform the subtraction to find change in density
merged['density_diff'] = merged['mean_myelin_5xFAD'] - merged['mean_myelin_6mos']
merged['abs_norm_diff'] = merged['density_diff']/merged['mean_myelin_6mos']

### Filter by depth of structure
# merged = merged[(merged['st_level'] > 5) & (merged['st_level'] < 9)]


# linear_regress(merged, sav_fold, exp_name + '_HIPPO', x='abs_norm_diff', y='density_W', figsize=(3.5, 3), xlim=1, fontsize=12)

# hippo_vals, hippo_names = get_subkeys_to_plot(df_means, FAD_df, reg_name='Hippocampal formation', dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=11)

# hippo_acros = hippo_vals['acronym'].tolist()  # from earlier hippocampal plot
linear_regress(merged, sav_fold, exp_name + '_ALLREGIONS', x='abs_norm_diff', y='density_W',
               figsize=(3.5, 3), 
                # xlim=(-1, 1), 
               ylim=(0, 5000),
               xlabel='Fold change myelin intensity', ylabel='Plaque Density',
               fontsize=12, annotate_all=False, 
               # highlight_acronyms=hippo_acros
               annotate_top_n=20
               )

    
zzz

#%% Plot average maps and then do centerline tracing of Fimbria
    

# Create group-wise storage
fimbria_by_group = defaultdict(list)

# Fill the dict
for id_f, fimbria in enumerate(fimbria_all):
    group = list_brains[id_f]['exp']  # or 'group' or whatever field holds '5xFAD', 'control', etc.
    
    # if list_brains[id_f]['num'] == 'M217':  ### skip 217 - older tissue - weird and dimmer?
    #     continue

    
    # if list_brains[id_f]['num'] == 'M243':  ### also shifted
    #     continue
    
    
    if list_brains[id_f]['num'] == 'M244':  ### very expanded
        continue
    
    

    # controls    
    # if list_brains[id_f]['num'] == 'M242':  ### shrunken fimbria
    #     continue
    # if list_brains[id_f]['num'] == 'M297':  ### misaligned and a little shrunken overall
    #     continue    

    fimbria_by_group[group].append(fimbria)    


### Make mean across all brains
# import napari
# viewer = napari.Viewer()


fimbria_avg_by_group = {}
for group, fimbria_list in fimbria_by_group.items():

    fimbria_stack = np.stack(fimbria_list, axis=0)  # shape: (N, Z, Y, X)
   
    # fimbria_stack = np.asarray(fimbria_stack, dtype=np.float32)  ### mask out out-of-bounds
    # fimbria_stack[fimbria_stack < 350] = np.nan
    


    fimbria_avg = np.nanmean(fimbria_stack, axis=0)  # nan-aware mean
    
    
    # fimbria_avg = np.asarray(fimbria_avg, dtype=np.uint16)
    # fimbria_avg = np.asarray(fimbria_avg, dtype=np.float32)   
    fimbria_avg_by_group[group] = fimbria_avg

    # viewer.add_image(fimbria_avg)
    
    # ### MASK OUT --- out of bounds and weirdness
    # fimbria_stack = np.asarray(fimbria_stack, dtype=np.float32)
    # fimbria_stack[fimbria_stack < 350] = np.nan
    # fimbria_nan = np.nanmean(fimbria_stack, axis=0)  # nan-aware mean
    # # fimbria_avg_by_group[group] = fimbria_avg
    
    # viewer.add_image(fimbria_nan)



# viewer.show(block=True)




# for group, fimbria_avg in fimbria_avg_by_group.items():
#     plot_max(fimbria_avg)
#     plt.title(f'{group} (avg fimbria)')


# Plot average maps
proj = plot_proj(
    im=fimbria_avg_by_group['6mos'],
    ax=0,
    method="mean",
    plot=True,
    vmin=0,
    vmax=1600,
    cmap="turbo",
    colorbar=True,
    cbar_label="Fluorescence"
)
plt.savefig(sav_fold + '_CONTROL_mean_myelin_smoothed.svg', format='svg', dpi=300)



proj = plot_proj(
    im=fimbria_avg_by_group['5xFAD'],
    ax=0,
    method="mean",
    plot=True,
    vmin=0,
    vmax=1600,
    cmap="turbo",
    colorbar=True,
    cbar_label="Fluorescence"
)
plt.savefig(sav_fold + '_5xFAD_mean_myelin_smoothed.svg', format='svg', dpi=300)






# Apply Gaussian smoothing to each group's mean map
sigma = 2.0  # in voxels; adjust depending on your resolution
smoothed_5xfad = gaussian_filter(fimbria_avg_by_group['5xFAD'], sigma=sigma)
smoothed_control = gaussian_filter(fimbria_avg_by_group['6mos'], sigma=sigma)


# Subtract the smoothed maps
diff_map = smoothed_control - smoothed_5xfad





#%% Fimbria centerline extraction

# left_atlas = np.copy(ref_atlas)
# left_atlas[:, :, 0:ref_atlas.shape[2] // 2] = 0

# Step 1: Get all subregion IDs for fimbria
# sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='fimbria')
# fimbria_ids = keys_df.iloc[sub_idx]['ids'].values
# Step 2: Build mask where atlas has one of those IDs
# fimbria_mask = np.isin(left_atlas, fimbria_ids)

left_fimbria = np.copy(fimbria_mask)
left_fimbria[:, :, 0:ref_atlas.shape[2] // 2] = 0

fimbria_mask = left_fimbria
# 1. Skeletonize
skeleton = skeletonize_3d(fimbria_mask)

# 2. Provide approximate endpoints (from your anatomical intuition)
approx_start = (260, 192, 301)  # e.g., dorsal/rostral end
approx_end = (381, 248, 423)    # e.g., ventral/caudal end

# 3. Extract the path
centerline_coords = extract_centerline_from_endpoints(skeleton, approx_start, approx_end)


def extend_centerline_to_anchor(centerline_coords, final_target, max_length=30):
    """
    Extend the centerline to a specified endpoint using a linear path.

    Parameters:
    - centerline_coords: (N, 3) array of current centerline points
    - final_target: (x, y, z) coordinate to extend toward
    - max_length: max number of voxels to extend

    Returns:
    - extended_coords: (N+M, 3) array with extension
    """
    from scipy.interpolate import interp1d
    from skimage.draw import line_nd

    end = centerline_coords[0]
    distance = np.linalg.norm(np.array(final_target) - end)
    n_steps = int(min(max_length, np.ceil(distance)))
    
    if n_steps < 2:
        return centerline_coords  # already close enough

    line_pts = np.linspace(final_target, end, n_steps).astype(int)
    return np.vstack([line_pts[1:], centerline_coords])


# Step 2: Extend it to your anatomical endpoint --- adds a few extra voxels to ANTERIOR side --- note ordering of points matters!!!
final_target = (280, 175, 316)
centerline_extended = extend_centerline_to_anchor(centerline_coords, final_target, max_length=30)

centerline_coords = centerline_extended

def remove_consecutive_duplicates(coords):
    # Create a mask to keep the first point and any point different from the previous
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(np.diff(coords, axis=0), axis=1)
    return coords[keep]

centerline_coords = remove_consecutive_duplicates(centerline_coords)


# 4. Convert to binary mask
centerline_mask = np.zeros_like(fimbria_mask, dtype=bool)
centerline_mask[tuple(centerline_coords.T)] = True




# import napari
# viewer = napari.Viewer()
# viewer.add_image(fimbria_mask)
# # viewer.add_image(skeleton)
# viewer.add_image(centerline_mask)
# viewer.show(block=True)


#%% Transform RAW intensity to line using centerline
def plot_straightened_projection_voxel_scaled_mm_flipped_y(
    projection_2d,
    voxel_size_um=20,
    cmap="turbo",
    vmin=0,
    vmax=1500,
    title="Straightened Fimbria Projection",
    cbar_label="Fluorescence Intensity",
    fontsize=12
):
    """
    Plot a straightened 2D fimbria projection with mm axes and origin at bottom-left.

    Parameters:
    - projection_2d: 2D array (Length, Width), already straightened
    - voxel_size_um: voxel resolution in microns (default 20)
    - cmap: colormap for imshow
    - vmin, vmax: intensity limits for imshow and colorbar
    - title: title string for plot
    - cbar_label: label for the colorbar
    - fontsize: base font size for all text
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(projection_2d.T, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax, origin='lower')

    # Axis labels (mm)
    ax.set_xlabel("Position along fimbria centerline (mm)", fontsize=fontsize)
    ax.set_ylabel("Transverse axis (mm)", fontsize=fontsize)
    # ax.set_title(title, fontsize=fontsize)

    # X-ticks (length)
    total_length_mm = projection_2d.shape[0] * voxel_size_um / 1000
    xtick_spacing = 1.0
    xtick_mm = np.arange(0, total_length_mm + xtick_spacing, xtick_spacing)
    xtick_vox = (xtick_mm * 1000 / voxel_size_um).astype(int)
    xtick_vox = xtick_vox[xtick_vox < projection_2d.shape[0]]
    ax.set_xticks(xtick_vox)
    ax.set_xticklabels(np.round(xtick_vox * voxel_size_um / 1000, 2), fontsize=fontsize-2)

    # Y-ticks (width)
    total_width_mm = projection_2d.shape[1] * voxel_size_um / 1000
    ytick_spacing = 0.5
    ytick_mm = np.arange(0, total_width_mm + ytick_spacing, ytick_spacing)
    ytick_vox = (ytick_mm * 1000 / voxel_size_um).astype(int)
    ytick_vox = ytick_vox[ytick_vox < projection_2d.shape[1]]
    ax.set_yticks(ytick_vox)
    ax.set_yticklabels(np.round(ytick_vox * voxel_size_um / 1000, 2), fontsize=fontsize-2)

    # Optional: re-enable colorbar if needed
    # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
    #                     ticks=np.linspace(vmin, vmax, 6))
    # cbar.set_label(cbar_label, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()



projection_2d_control = straighten_fimbria_with_z_projection(
    image=smoothed_control,
    mask=fimbria_mask,
    centerline_coords=centerline_coords,
    width=60,
    height=30,
    n_steps=len(centerline_coords),
    agg_func = 'mean',
)

# Make cleaned mask
fimbria_mask_2d, largest = clean_projection_with_mask(projection_2d_control, erosion_iter=2, dilation_iter=3)

# mask out
projection_2d_control[fimbria_mask_2d == 0] = 0
projection_2d_control[np.isnan(projection_2d_control)] = 0



plot_straightened_projection_voxel_scaled_mm_flipped_y(
    projection_2d=projection_2d_control,
    voxel_size_um=20,
    cmap="turbo",
    vmin=0,
    vmax=1500,
    title="Control 2D Z-max fimbria projection",
    cbar_label="Fluorescence Intensity",
    fontsize=14
)


plt.savefig(sav_fold + '_CONTROL_2D_straightened.svg', format='svg', dpi=300)




#%% Plot the diff map in myelin intensity straightened

projection_2d_diff = straighten_fimbria_with_z_projection(
    image=diff_map,
    mask=fimbria_mask,
    centerline_coords=centerline_coords,
    width=60,
    height=30,
    n_steps=len(centerline_coords),
    agg_func = 'mean',
)



# mask out
projection_2d_diff[fimbria_mask_2d == 0] = 0
projection_2d_diff[np.isnan(projection_2d_diff)] = 0


plot_straightened_projection_voxel_scaled_mm_flipped_y(
    projection_2d=projection_2d_diff,
    voxel_size_um=20,
    cmap="coolwarm",
    vmin=-75,
    vmax=75,
    title="Differential mean myelin intensity map",
    cbar_label="Intensity diff",
    fontsize=14
)
plt.savefig(sav_fold + '_2D_diffmap.svg', format='svg', dpi=300)





#%% Project halos ontop of all of this --- first convert halo coordinates to reference frame of atlas


all_dfs = []
for exp in list_brains:
    
    
    if exp['exp'] != '5xFAD':
        continue
    print('Loading coords_df from ' + exp['name'])
    
    keys_df = main_keys.copy(deep=True)

    input_path = exp['path']
    name_str = exp['name']
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'

    filename = n5_file.split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename) 
    sav_dir = input_path + '/' + filename + '_postprocess_CONGO'

    import pickle as pkl
    
    # Load pickle
    with open(sav_dir + '/coords_df_HALO_data', 'rb') as file:
        coords_df = pkl.load(file)
    
        
    coords_df['exp'] = exp['num']
    
    all_dfs.append(coords_df)


coords_df = pd.concat(all_dfs)




# downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
# atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'

# ### Convert coordinates into ATLAS reference frame
# downsampled_points = np.asarray([coords_df['Z_down'], coords_df['X_down'], coords_df['Y_down']]).T
# # and then move columns (1 to 0)  --> all these steps just do the inverse of original steps
# downsampled_points[:, [1, 0, 2]] = downsampled_points[:, [0, 1, 2]]

# deformation_field_paths = [atlas_dir + 'deformation_field_0_ADD_CERE.tiff',
#                             atlas_dir + 'deformation_field_1_ADD_CERE.tiff',
#                             atlas_dir + 'deformation_field_2_ADD_CERE.tiff'
#                           ]


# deformation_field = tiff.imread(deformation_field_paths[0])

# ### Flip Z axis
# # downsampled_points[:, 0] = deformation_field.shape[0] - downsampled_points[:, 0]
# downsampled_points[:, 1] = deformation_field.shape[1] - downsampled_points[:, 1]
# downsampled_points[:, 2] = deformation_field.shape[2] - downsampled_points[:, 2]
        

atlas_resolution = [20, 20, 20]

# field_scales = [int(1000 / resolution) for resolution in atlas_resolution]
# points = [[], [], []]
# for axis, deformation_field_path in enumerate(deformation_field_paths):
#     deformation_field = tiff.imread(deformation_field_path)
#     print('hello')
#     for point in downsampled_points:
#         point = [int(round(p)) for p in point]
        
    
#         points[axis].append(
#             ### REMOVED ROUNDING - TIGER --- for Congo analysis
#             #int(
#             #    round(
#                     field_scales[axis] * deformation_field[point[0], point[1], point[2]]
#             #    )
#             #)
#         )
        
# points = np.transpose(points)
# # p = np.copy(points)

# p = np.copy(np.round(points).astype(int))

# p  = np.clip(p , a_min=0, a_max=np.array(ref_atlas.shape) - 1)  ### ensure doesn't round out of bounds

# coords_df['dim0'] = p[:, 0]
# coords_df['dim1'] = p[:, 1]
# coords_df['dim2'] = p[:, 2]

    


#%% Get halos from fimbria 
fimbria_halos = coords_df[coords_df['region_group'].isin(['fimbria'])]
halo_centroids = fimbria_halos[['dim0', 'dim1', 'dim2']].values


# Remove padding and scale
centerline_coords_real = centerline_coords #- np.array([50, 50, 50])  # remove atlas padding
 
### Find out which halos are within the fimbria - on only the masked side!!!
matched_halo_ids = np.where(fimbria_mask[halo_centroids[:, 0], halo_centroids[:, 1], halo_centroids[:, 2]] > 0)[0]

halo_centroids = halo_centroids[matched_halo_ids]
# halo_centroids_ATLAS = halo_voxels[matched_halo_ids]
halo_mask = np.zeros(np.shape(fimbria_mask))
halo_mask[halo_centroids[:, 0], halo_centroids[:, 1], halo_centroids[:, 2]] = 1    
    

### Make linear projection onto centerline
halo_indices_along_fimbria = project_points_to_centerline(halo_centroids, centerline_coords_real)






#%% Make additive DENSITY of halos plot

radius = 8

additive_density = build_additive_halo_density_volume(
    halo_voxels=halo_centroids,  # voxel-space coordinates
    shape=fimbria_mask.shape,
    radius=radius  # adjust based on voxel resolution (e.g. 5 voxels = 100 µm radius)
)

### Exclude regions with no tissue
additive_density[ref_atlas == 0] = 0

### scale to cells / mm^3  (currently is in cells/0.064 mm^3) ---> so multiply image by 

cur_volume = ((radius * atlas_resolution[0] * 0.001) * 2)**3  ### in cells/mm^3
scale_factor = 1/cur_volume    ### now scaled to cells/1 mm^3


additive_density = additive_density * scale_factor

### save density image
# tiff.imwrite(sav_dir + exp['name'] + '_DENSITY_MAP_CERE_rad5_minsize30.tif', np.asarray(sphere_im, dtype=np.uint32))
        

projection_2d_density = straighten_fimbria_with_z_projection(
    image=additive_density,
    mask=fimbria_mask,
    centerline_coords=centerline_coords,
    width=60,
    height=30,
    n_steps=len(centerline_coords),
    agg_func = 'mean',
)


# mask out
projection_2d_density[fimbria_mask_2d == 0] = 0
projection_2d_density[np.isnan(projection_2d_density)] = 0



plot_straightened_projection_voxel_scaled_mm_flipped_y(
    projection_2d=projection_2d_density,
    voxel_size_um=20,
    cmap="turbo",
    vmin=0,
    vmax=1200,
    title="Halo density additive",
    cbar_label="Halo density (1/mm$^3$)",
    fontsize=14
)

plt.savefig(sav_fold + '_2D_projected_halo_density_additive.svg', format='svg', dpi=300)






#%% Also make a plot of halo loss per length bin --- SHARED y-axis 1d projection
fimbria_halos['percent_change'] = (fimbria_halos['c1_myelin'] - fimbria_halos['myelin_int'])/fimbria_halos['c1_myelin'] * 100

fimbria_right_halos = fimbria_halos.copy()
fimbria_right_halos = fimbria_right_halos.iloc[matched_halo_ids]


# Get halo positions along centerline
halo_pos = project_points_to_centerline(halo_centroids, centerline_coords_real)
n_bins = 20
centerline_len = len(centerline_coords_real)

# 1. Binned fold change
fc_vals = fimbria_right_halos['percent_change'].values
bin_fc_x, bin_fc_y, bin_fc_sem = bin_along_centerline(fc_vals, halo_pos, n_bins, centerline_len)
# 2. Binned additive density (projected 2D → 1D → binned)
density_1d = np.nanmean(np.where(fimbria_mask_2d, projection_2d_density, np.nan), axis=1)
bin_density_x, bin_density_y, bin_density_sem = bin_along_centerline(density_1d, np.arange(centerline_len), n_bins, centerline_len)






# Colors from Set2
color_fc = palette[0]  # usually green
color_density = palette[1]  # usually orange

# fig, ax1 = plt.subplots(figsize=(8, 3))

# # Plot fold change (left axis)
# ax1.errorbar(bin_fc_x * atlas_resolution[0] * 0.001, bin_fc_y, yerr=bin_fc_sem,
#              color=color_fc, marker='o', linestyle='-', label='Myelin % Change', #ecolor='gray', 
#              markersize=5, capsize=3)
# ax1.set_xlabel("Position Along Fimbria", fontsize=fontsize)
# ax1.set_ylabel("Mean myelin loss per halo (%)", fontsize=fontsize, color=color_fc)
# ax1.tick_params(axis='y', labelcolor=color_fc, labelsize=ticksize)
# ax1.tick_params(axis='x', labelsize=ticksize)
# ax1.axhline(0, color='k', linestyle='--', linewidth=1)
# ax1.set_ylim([-5, 50])

# # Plot halo density (right axis)
# ax2 = ax1.twinx()
# ax2.errorbar(bin_density_x * atlas_resolution[0] * 0.001, bin_density_y, yerr=bin_density_sem,
#              color=color_density, marker='s', linestyle='-', label='Halo Density', #ecolor='gray',
#              markersize=5, capsize=3)
# ax2.set_ylabel("Halo Density (1/mm$^3$)", fontsize=fontsize, color=color_density)
# ax2.tick_params(axis='y', labelcolor=color_density, labelsize=ticksize)
# ax2.set_ylim([0, 500])

# # Spine and layout cleanup
# ax1.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)

# plt.xlim([0, 4.5])

# plt.xticks(np.arange(0, bin_fc_x[-1] * atlas_resolution[0] * 0.001 + 0.5, 0.5), fontsize=ticksize)

# # plt.title("Halo and Myelin Profiles Along Fimbria", fontsize=fontsize)
# plt.tight_layout()
# plt.show()


# plt.savefig(sav_fold + '_loss_per_halo_and_halo_density.svg', format='svg', dpi=300)








# #%% Plot binned ABSOLUTE number of plaques along fimbria
# color_counts = palette[2]  # Consistent Set2 color for counts (e.g., pink/red)
# # Step 1: Bin halo indices
n_bins = 20
centerline_len = len(centerline_coords_real)
bins = np.linspace(0, centerline_len - 1, n_bins + 1)
bin_indices = np.digitize(halo_indices_along_fimbria, bins) - 1

# # Step 2: Count plaques per bin
counts = np.array([(bin_indices == i).sum() for i in range(n_bins)])
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# # Step 3: Styled plot
# plt.figure(figsize=(8, 3))
# plt.plot(bin_centers * atlas_resolution[0] * 0.001, counts, color=color_counts, marker='o', linestyle='-', markersize=5)

# plt.xlabel("Position Along Fimbria", fontsize=fontsize)
# plt.ylabel("Plaque Count", fontsize=fontsize)
# # plt.title("Plaque Distribution Along Fimbria", fontsize=fontsize)

# plt.xticks(fontsize=ticksize)
# plt.yticks(fontsize=ticksize)
# plt.ylim([0, 60])
# plt.xlim([0, 4.5])

# plt.xticks(np.arange(0, bin_centers[-1] * atlas_resolution[0] * 0.001 + 0.5, 0.5), fontsize=ticksize)

# # Clean up spines
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.tight_layout()

# # Optional: Save as vector graphic
# # plt.savefig("plaque_count_along_fimbria.svg", dpi=300)

# plt.show()

# plt.savefig(sav_fold + '_1d_absolute_plaque_number_along_length.svg', format='svg', dpi=300)




# Modified version of the first figure:
# Halo density and absolute halo count share the same axes
xlim = 4.0
fig, ax1 = plt.subplots(figsize=(8, 3))

# Plot halo density (left axis)
ax1.errorbar(bin_density_x * atlas_resolution[0] * 0.001, bin_density_y, yerr=bin_density_sem,
             color='k', marker='s', linestyle='-', label='Halo density',
             markersize=5, capsize=3)
ax1.set_xlabel("Position along fimbria", fontsize=fontsize)
ax1.set_ylabel("Halo density (1/mm$^3$)", fontsize=fontsize, color='k')
ax1.tick_params(axis='y', labelcolor='k', labelsize=ticksize)
ax1.tick_params(axis='x', labelsize=ticksize)
ax1.set_ylim([0, 800])

# Create right axis for absolute plaque count
# ax2 = ax1.twinx()
# ax2.plot(bin_centers * atlas_resolution[0] * 0.001, counts,
#          color=color_density, marker='o', linestyle='-', label='Plaque count',
#          markersize=5)
# ax2.set_ylabel("Plaque count", fontsize=fontsize, color=color_density)
# ax2.tick_params(axis='y', labelcolor=color_density, labelsize=ticksize)
# ax2.set_ylim([0, 60])

# Spine and layout cleanup
ax1.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.xlim([0, xlim])
plt.xticks(np.arange(0, bin_fc_x[-1] * atlas_resolution[0] * 0.001 + 0.1, 0.5), fontsize=ticksize)

# plt.title("Plaque Density and Count Along Fimbria", fontsize=fontsize)
plt.tight_layout()
plt.show()

plt.savefig(sav_fold + '_halo_density_and_count.svg', format='svg', dpi=300)


# Modified version of the second figure:
# Myelin loss per halo moved here as a standalone plot

plt.figure(figsize=(8, 3))
plt.errorbar(bin_fc_x * atlas_resolution[0] * 0.001, bin_fc_y, yerr=bin_fc_sem,
             color=color_fc, marker='o', linestyle='-', label='Myelin % Change',
             markersize=5, capsize=3)
plt.xlabel("Position along fimbria", fontsize=fontsize)
plt.ylabel("Mean myelin loss per halo (%)", fontsize=fontsize, color=color_fc)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.ylim([-5, 50])
plt.xlim([0, xlim])
plt.axhline(0, color='k', linestyle='--', linewidth=1)

plt.xticks(np.arange(0, bin_fc_x[-1] * atlas_resolution[0] * 0.001, 0.5), fontsize=ticksize)

# Clean up spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.title("Mean Myelin Loss per Halo Along Fimbria", fontsize=fontsize)
plt.tight_layout()
plt.show()

plt.savefig(sav_fold + '_loss_per_halo_along_length.svg', format='svg', dpi=300)







#%% Check our work - which side is dorsal/ventral ect...
def plot_reference_with_centerline_and_edges(
    max_proj,
    centerline_coords,
    projection_2d,
    edge_index=None,
    voxel_size_um=20
):
    """
    Plot max projection of reference volume with centerline and tangential edge markers.

    Parameters:
    - max_proj: 2D image from np.max(image, axis=0), shape (Z, Y)
    - centerline_coords: (N, 3) array in (X, Z, Y)
    - projection_2d: 2D array (Length, Width)
    - edge_index: index into centerline to use for edge visualization
    - voxel_size_um: voxel resolution (default 20)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if edge_index is None:
        edge_index = projection_2d.shape[0] * 4 // 5

    plt.figure()
    ax = plt.gca()
    ax.imshow(max_proj, cmap='Oranges', aspect='auto', origin='upper')
    ax.set_title("Reference Volume (X collapsed)")
    ax.set_xticks([])
    ax.set_yticks([])

    centerline_yz = centerline_coords[:, [2, 1]]  # Y, Z
    ax.plot(centerline_yz[:, 0], centerline_yz[:, 1], color='cyan', linewidth=1.5, label='Centerline')
    ax.scatter(centerline_yz[0, 0], centerline_yz[0, 1], color='lime', label='Start')
    ax.scatter(centerline_yz[-1, 0], centerline_yz[-1, 1], color='magenta', label='End')

    y_center = centerline_yz[edge_index, 0]
    z_center = centerline_yz[edge_index, 1]
    half_width = projection_2d.shape[1] // 2
    y_left = int(np.clip(y_center + half_width, 0, max_proj.shape[1] - 1))
    y_right = int(np.clip(y_center - half_width, 0, max_proj.shape[1] - 1))

    ax.scatter(y_right, z_center, color='blue', label='Right edge')
    ax.scatter(y_left, z_center, color='orange', label='Left edge')
    ax.plot([y_right, y_left], [z_center, z_center], linestyle='dashed', color='k', linewidth=1.5)

    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def plot_fimbria_projection_with_scaling(
    projection_2d,
    voxel_size_um=20,
    cmap="turbo",
    vmin=0,
    vmax=1500,
    title="Straightened Fimbria Projection",
    cbar_label="Fluorescence Intensity",
    edge_index=None
):
    """
    Plot straightened fimbria projection with mm scaling and flipped Y axis.

    Parameters:
    - projection_2d: 2D array (Length, Width)
    - voxel_size_um: voxel resolution (default 20)
    - cmap: colormap
    - vmin, vmax: intensity range
    - title: plot title
    - cbar_label: label for the colorbar
    - edge_index: optional vertical dashed line (e.g. to match tangent plane)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    im = ax.imshow(projection_2d.T, cmap=cmap, aspect='auto',
                   origin='lower', vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Length (mm)")
    ax.set_ylabel("Width (mm)")

    # X ticks
    xtick_spacing = 0.5
    total_length_mm = projection_2d.shape[0] * voxel_size_um / 1000
    xtick_mm = np.arange(0, total_length_mm + xtick_spacing, xtick_spacing)
    xtick_vox = (xtick_mm * 1000 / voxel_size_um).astype(int)
    xtick_vox = xtick_vox[xtick_vox < projection_2d.shape[0]]
    ax.set_xticks(xtick_vox)
    ax.set_xticklabels(np.round(xtick_vox * voxel_size_um / 1000, 2))

    # Y ticks
    ytick_spacing = 0.2
    total_width_mm = projection_2d.shape[1] * voxel_size_um / 1000
    ytick_mm = np.arange(0, total_width_mm + ytick_spacing, ytick_spacing)
    ytick_vox = (ytick_mm * 1000 / voxel_size_um).astype(int)
    ytick_vox = ytick_vox[ytick_vox < projection_2d.shape[1]]
    ax.set_yticks(ytick_vox)
    ax.set_yticklabels(np.round(ytick_vox * voxel_size_um / 1000, 2))

    if edge_index is not None:
        ax.axvline(edge_index, linestyle='dashed', color='magenta', linewidth=1.5)
        ax.scatter(edge_index, 0, color='blue', label='Right edge')
        ax.scatter(edge_index, projection_2d.shape[1] - 1, color='orange', label='Left edge')

    ax.scatter(0, projection_2d.shape[1] // 2, color='lime', label='Start')
    ax.scatter(projection_2d.shape[0] - 1, projection_2d.shape[1] // 2, color='magenta', label='End')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()



edge_index = 160  # or let it default

plot_reference_with_centerline_and_edges(
    max_proj=np.max(fimbria_mask, axis=0),
    centerline_coords=centerline_coords,
    projection_2d=projection_2d_control,
    edge_index=edge_index
)


plt.savefig(sav_fold + '_centerline_plotted.svg', format='svg', dpi=300)


plot_fimbria_projection_with_scaling(
    projection_2d=projection_2d_control,
    voxel_size_um=20,
    cmap="turbo",
    vmin=0,
    vmax=1500,
    title="Straightened Fimbria Projection",
    cbar_label="Fluorescence Intensity",
    edge_index=edge_index
)


plt.savefig(sav_fold + '_centerline_plotted_on_2D.svg', format='svg', dpi=300)




    
