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

import tifffile as tiff

import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean

from skimage.measure import label, regionprops, regionprops_table

import json
import matplotlib.pyplot as plt

### Makes it so that svg exports text as editable text!
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'    


from postprocess_func import *
import seaborn as sns
from scipy import stats   
from adjustText import adjust_text


import sys
sys.path.append("..")

from get_brain_metadata import *

from PLOT_FUNCTIONS_postprocess_compare import *
        

#%% List of brains

new_large_OL = 0

# list_brains = get_metadata(mouse_num = 'all')
# list_brains = get_metadata(mouse_num = ['M243', 'M297'])

list_brains = get_metadata(mouse_num = ['M217', 'M243', 'M244', 'M234', 'M235',     ### 5xFAD
                                        'M296', 'M297', 'M304', 'M242'
                                        ])    ### Controls 6mos



### For myelin intensity density comparison later --- M235 seems brighter, M217 seems more dim


if new_large_OL:
    list_brains = get_metadata(mouse_num = ['M260', 'M286'])

# print('CUPRIZONE AND OLD BRAIN (anything with lots of lipofuscin) currently using -10grid (only 1st old brain, all newer analyzed old brains using -15grid)')

sav_fold = '/media/user/4TB_SSD/Plot_outputs_CONGO/'


# myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
# auto_path = glob.glob(os.path.join(downsampled_dir, '*_ch1_PAD.tif'))[0]

pad = True

XY_res = 1.152035240378141
Z_res = 5
res_diff = XY_res/Z_res

ANTS = 1

#%%%% Parse the json file so we can choose what we want to extract or mask out
with open('../atlas_ids/atlas_ids.json') as json_file:
    data = json.load(json_file)
 
     
data = data['msg'][0]

keys_tmp = get_ids_all(data, all_keys=[], keywords=[''])  
keys_tmp = pd.DataFrame.from_dict(keys_tmp)


#%%% REMOVE ADDITIONAL TORN TISSUE REGIONS - as specified in metadata (including replacements if given hemisphere)
def additional_preprocess(keys_df, info):
    drop_ids = []
    if 'exclude' in info:
        print('removing torn regions')
        # zzz
        
        exclude = info['exclude']
        

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
                keys_df['num_OLs_L'] = keys_df['num_OLs_' + exclude[0][1]]
                keys_df['num_OLs_R'] = keys_df['num_OLs_' + exclude[0][1]]
                keys_df['num_OLs_W'] = keys_df['num_OLs_' + exclude[0][1]] * 2  
        
        
        ### Do by region
        else:
            all_ids = []
            for region in exclude:
                
                
                reg_ids = get_sub_regions_by_acronym(keys_df, child_id=[], sub_keys=[], reg_name=region[0])
                
                ### if we decide to only numbers from one hemisphere, then overwrite data of other hemisphere
                if region[1] != 'B' and region[1] != '':
                    
                    keys_df.loc[reg_ids, 'num_OLs_L'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]]
                    keys_df.loc[reg_ids, 'num_OLs_R'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]]
                    keys_df.loc[reg_ids, 'num_OLs_W'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]] * 2
        
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
            
            df_sum = df_sum.drop(['ids', 'parent', 'st_level', 'density_W', 'age'])
            
            row[df_sum.index] = df_sum
            
            
            df_level.loc[i] = row  ### add row back in
          
    keys_df = df_level
    
    if 'side' in info:
        
        keys_df['density_W'] = keys_df['num_OLs_' + info['side']]/keys_df['atlas_vol_' + info['side']]
        keys_df['density_LARGE_W'] = keys_df['num_large_' + info['side']]/keys_df['atlas_vol_' + info['side']] 
        
    else:
        keys_df['density_L'] = keys_df['num_OLs_L']/keys_df['atlas_vol_L']
        keys_df['density_R'] = keys_df['num_OLs_R']/keys_df['atlas_vol_R']
        keys_df['density_W'] = keys_df['num_OLs_W']/keys_df['atlas_vol_W']
        

        if 'num_large_W_CLEAN' in keys_df.keys():
            keys_df['density_LARGE_W_CLEAN'] = keys_df['num_large_W_CLEAN']/keys_df['atlas_vol_W']
        else:
            keys_df['density_LARGE_W'] = keys_df['num_large_W']/keys_df['atlas_vol_W']




    return keys_df






#%%%% Parse pickles from all folders
""" Loop through all the folders and pool the dataframes!!!"""

all_coords_df = []
all_coords_df_CONGO = []
all_keys_df = []
all_keys_df_CONGO = []
fold_names = []
for id_f, info in enumerate(list_brains):

    fold = info['path'] + info['name'] + '_postprocess/'
    fold_CONGO = info['path'] + info['name'] + '_postprocess_CONGO/' 
    
    
    fold_names.append(info['name'])
    

    pkl_to_use = info['pkl_to_use']

    if not ANTS:
        keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid.pkl'))    
    else:
        # keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY.pkl'))   
        
        if pkl_to_use == 'MYELIN':
            keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl'))   
            
 
        elif pkl_to_use == 'CUBIC':
            keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE_CUBIC.pkl')) 
            
            
            
    coords_df = pd.read_pickle(fold + info['name'] + '_coords_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl')  
    coords_df['exp_name'] = info['name']
    all_coords_df.append(coords_df)
            

    

           
    try:    
        keys_df = pd.read_pickle(keys[0])
        
        # keys_df['congo'] = False
    except:
        print('Missing: ' + fold)
        continue
    # coords_df = pd.read_pickle(coords[0])
    
    # print('number of cells: ' + str(len(coords_df)))
    
    
    
    
    # ### if congo keys_df exists then load it as well
    if os.path.isfile(fold_CONGO + info['name'] + '_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl'):
        print('Loading congo')
        keys_congo = pd.read_pickle(fold_CONGO + info['name'] + '_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl')    
        
        coords_congo = pd.read_pickle(fold_CONGO + info['name'] + '_coords_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl')    
    
        coords_congo['exp_name'] = info['name']
            
          
        all_coords_df_CONGO.append(coords_congo)
        
        ### append to keys_df???
        # keys_congo['congo'] = True
        keys_congo['acronym'] = keys_tmp['acronym']
        keys_congo['dataset'] = info['num']
        keys_congo['exp'] = info['exp']
        keys_congo['sex'] = info['sex']
        keys_congo['age'] = info['age']   
        
        keys_congo = additional_preprocess(keys_congo, info)
        all_keys_df_CONGO.append(keys_congo)
        
    
    keys_df['acronym'] = keys_tmp['acronym']
    keys_df['dataset'] = info['num']
    keys_df['exp'] = info['exp']
    keys_df['sex'] = info['sex']
    keys_df['age'] = info['age']   
    
    # keys_df = keys_df.reset_index()
    keys_df = additional_preprocess(keys_df, info)
    all_keys_df.append(keys_df)
    


df_concat = pd.concat(all_keys_df)

df_congo = pd.concat(all_keys_df_CONGO)

exp_name = 'compare_CONGO'
exp_name = 'compare_control'

### Exclude M229 --- mostly just for large OL counting since stitching was only translational
### Included for now... just for fun
# aging_df = df_concat[~df_concat['dataset'].isin(['M229'])]


import pickle

# Save to pickle
with open(sav_fold + 'df_congo.pkl', 'wb') as f:
    pickle.dump(df_congo, f)

zzz


#%% LARGE OL pooling:
if exp_name == 'LARGE' or plot_all:
    exp_name = 'LARGE_5xFAD'

    ### Declare plotting variables
    palette = sns.color_palette("Set2")
    
    AD_df_LARGE = df_concat
    
    AD_df_LARGE = AD_df_LARGE[AD_df_LARGE['exp'].isin(['5xFAD', '6mos'])]
    
    df_means = AD_df_LARGE.groupby(AD_df_LARGE.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
       
        
    




    
    fontsize = 12
    # exp_name = '5xFAD'
    volcano_compare(AD_df, keys_df, compareby='exp', group1='6mos', group2='5xFAD',   ### group2 - group1 for means (so up is group2 > group1)
                    xplot='log2fold', 
                    compare_col='density_LARGE_W_CLEAN',
                    parcel_lvl_limit=9,   ### max (normally 9)
                    
                    thresh_pval=0.05, thresh_log2=0.3,
                    # thresh_pval=0.01, thresh_log2=0.3,  ### original threshold values
                    fontsize=fontsize, xlim=1, ylim=6, figsize=(3.2, 3))
    
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE_LARGE.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE_LARGE.svg', format='svg', dpi=300)


    
    
    
    
    #%%% PLOT global regions of interest

    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', #'AUDd', 'AUDpo', 'AUDv', 
                     'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', #'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(AD_df_LARGE, names_to_plot, palette,  sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX',
                   dname='density_LARGE_W_CLEAN', dropna=True,
                   ylim=300, figsize=(7, 2.6))
    
    
    names_to_plot = [
                     'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg',   ### Hippocampus
                      'ENT', 'ENTl', 'ENTm', #'ENTmv',    ### Entorhinal areas divided into layer 1 - 6!!!
                     ]

    plot_global(AD_df_LARGE, names_to_plot, palette, sav_fold,  exp_name + '_GLOBAL_COMPARISON_HIPPO',
                   dname='density_LARGE_W_CLEAN', dropna=True,
                   ylim=300, figsize=(4, 3))
    
    


    
    
    #%%% Get rank order of areas with highest density of large OLs in old  brains!
    
    df_means = AD_df_LARGE.groupby(AD_df_LARGE.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
        

    def plot_rank_order(plot_vals, exp_name, dname, figsize=(4,4), fontsize=14):
        
        old_rate = plot_vals[plot_vals['exp'].str.fullmatch(exp_name) == True]  ### get only P620 brains
    
        old_rate = old_rate[old_rate[dname].notna()]
        
        plot_order = old_rate.groupby(['acronym']).mean(numeric_only=True).reset_index().sort_values(by=dname, ascending=False)
        
        plt.figure(figsize=figsize)
        sns.barplot(old_rate, x=dname, y='acronym', order=plot_order['acronym'], color='grey',
                    errorbar='se')   
        
        ax = plt.gca()
        #sns.move_legend(ax, loc=leg_loc, frameon=False, title='', fontsize=fontsize)
        plt.yticks(fontsize=fontsize - 2)
        plt.xticks(fontsize=fontsize - 2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylabel('', fontsize=fontsize)
        plt.xlabel('Density of large OLs (cells/mm\u00b3)', fontsize=fontsize)
        plt.tight_layout()
    
        plt.savefig(sav_fold + exp_name +'_LARGE_OLs_rank_order.png', format='png', dpi=300)
        plt.savefig(sav_fold + exp_name +'_LARGE_OLs_rank_order.svg', format='svg', dpi=300)
        

    
    to_remove='VISrl|AUDd|VISal|VISl|AUDpo|VISa|AUDv|VISam|VISli|VISpm|VISpl|VISpor|MO|SS|AUD|VIS'
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, AD_df_LARGE, reg_name='Isocortex', dname='density_LARGE_W_CLEAN', 
                                                    lvl_low=5, lvl_high=9, to_remove=to_remove)
    
    plot_rank_order(plot_vals, exp_name='6mos', dname='density_LARGE_W_CLEAN', 
                    figsize=(3.5,4))    
    
    plot_rank_order(plot_vals, exp_name='5xFAD', dname='density_LARGE_W_CLEAN', 
                    figsize=(3.5,4))    
    

    
        
    
    
    

    
    
    
    ### Plot regression
    
    # to_remove = 'VISpl'
    # plot_vals, names_to_plot = get_subkeys_to_plot(df_means, AD_df_LARGE, reg_name='Isocortex', dname='density_LARGE_W_CLEAN', to_remove=to_remove, lvl_low=5, lvl_high=9)
    
    # all_avg = []
    # for i_n, reg_name in enumerate(names_to_plot):
    #     match = plot_vals[plot_vals['acronym'].str.fullmatch(reg_name) == True]
    
    #     # baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
    #     # match['fold_change'] = match['density_W']/baseline
        
    #     ### Also relate beginning density to fold change?
    #     avg_stats = match.groupby(['exp', 'acronym']).mean(numeric_only=True).reset_index()
    #     # copy the density of LncOL1 cells over so can plot altogether later!
    #     avg_stats.loc[np.where(avg_stats['exp'] == '6mos')[0], 'density_W'] = avg_stats.loc[np.where(avg_stats['exp'] == '5xFAD')[0], 'density_W'].values[0]
         
    #     avg_stats = avg_stats.loc[np.where(avg_stats['exp'] == '6mos')[0]]
        
    #     all_avg.append(avg_stats)
    
    
    # all_avg = pd.concat(all_avg)
        
    
    
        
    # plt.figure(figsize=(4,3))
    # fontsize=12
    # fig = sns.regplot(data=all_avg, x='density_W', 
    #                                 y='density_LARGE_W_CLEAN', color='gray')

    # r,p = stats.pearsonr(all_avg.dropna()['density_W'], all_avg.dropna()['density_LARGE_W_CLEAN'])
    # print(r)
    
    
    # # vals = stats.linregress(all_avg.dropna()['density_W'], all_avg.dropna()['density_LARGE_W'])  ### THIS CAN ALSO GIVE YOU SLOPE
    
    
    # all_texts = []
    # for line in range(0,all_avg.shape[0]):
    #      all_texts.append(plt.text(all_avg['density_W'].iloc[line]+0.01, all_avg['density_LARGE_W_CLEAN'].iloc[line], 
    #      all_avg['acronym'].iloc[line],
    #      ha='center', va='center',
    #      size=10, color='black', weight='normal'))
    # #plt.text(all_texts)
    
    # ax = plt.gca()
    # ax.legend().set_visible(False)
    # plt.yticks(fontsize=fontsize - 2)
    # plt.xticks(fontsize=fontsize - 2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # #ax.spines['left'].set_visible(False)
    # #plt.legend(loc = 'upper left', frameon=False)
    # # plt.ylim([0, 30])
    # plt.xlabel('Density of LncOL1+ cells (cells/mm\u00b3)', fontsize=fontsize)
    # plt.ylabel('Density of large OLs (cells/mm\u00b3)', fontsize=fontsize)
    # plt.tight_layout()
    
    # adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    # plt.savefig(sav_fold + exp_name + '_density_P60_vs_LncOL1.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name + '_density_P60_vs_LncOL1.svg', format='svg', dpi=300)
    
    
    

    
    
    
    
    
    
    

#%% Compare between plaque and oligos and myelin density
if exp_name == 'compare_CONGO':
    
    congo_df = df_congo
    congo_df = congo_df[congo_df['exp'].isin(['5xFAD'])]
    df_means = congo_df.groupby(congo_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
          
    
    FAD_df = df_concat
    FAD_df = FAD_df[FAD_df['exp'].isin(['5xFAD'])]
    df_means = FAD_df.groupby(FAD_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
       
    
    ### Append congo density straight to FAD_df
    FAD_df['congo_density'] = congo_df['density_W'].values
    
    
    
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe
    
                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))
    
        
    plot_global(congo_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX',
                   ylim=2000, figsize=(7, 3))
        

    
    names_to_plot = [
                     'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg',   ### Hippocampus
    
                        ### Retrohippocampal regions
                      #'RHP', 
                      'ENT', 'ENTl', 'ENTm', #'ENTmv',    ### Entorhinal areas divided into layer 1 - 6!!!
                      # 'ENTl1', 'ENTl2', 'ENTl3', 'ENTl5', 'ENTl6a',
                      # 'ENTm1', 'ENTm2', 'ENTm3', 'ENTm5', 'ENTm6',
                      
                        'PAR', #'PRE',  
                      
                      ### Para- post- and pre-subiculum (also divided into layer 1, 2, 3!!!)
                      
                      
                      ### Areas below here are too high to fit on ylim... need 40000
                        # 'POST',
                        'ProS', 
                        'SUB', #'SUBv', 'SUBd',   ### subiculum
                       # 'fornix system',
    
                     ]
    
    plot_global(congo_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_HIPPO',
                   ylim=5000, figsize=(4.2, 2.5))




    #%% Find relationship between plaque load and OL density
    def linear_regress(plot_vals_raw, sav_fold, exp_name, x='density_W', y='congo_density', figsize=(3.5, 3), xlim=None, ylim=None, xlabel=None, ylabel=None, fontsize=12, annotate_all=True, highlight_acronyms=None, annotate_top_n=None):
        
        # Filter out rows where plaque density is zero
        plot_vals = plot_vals_raw[plot_vals_raw[y] != 0].copy()
    
    
        plt.figure(figsize=figsize)
        marker_size = 20 if len(plot_vals) <= 500 else 8
        # fig = sns.regplot(data=plot_vals, x=x, y=y, scatter_kws={'s': marker_size}, line_kws={'color': '#00A676'}, color='gray')
        fig = sns.regplot(data=plot_vals, x=x, y=y, scatter_kws={'s': marker_size}, line_kws={'color': '#D62728'}, color='gray')
      
        # calculate slope, intercept, correlation
        slope, intercept, r_val, p_val, sterr = stats.linregress(
            x=fig.get_lines()[0].get_xdata(),
            y=fig.get_lines()[0].get_ydata()
        )
    
        r, p = stats.pearsonr(plot_vals.dropna()[x], plot_vals.dropna()[y])
        # print(f"slope: {slope:.4f}, r: {r:.4f}, p: {p:.4e}")
    
        # Spearman correlation
        rho, spearman_p = stats.spearmanr(plot_vals.dropna()[x], plot_vals.dropna()[y])
    
        # Print results
        print(f"slope: {slope:.4f}, Pearson r: {r:.4f}, p: {p:.4e}")
        print(f"Spearman rho: {rho:.4f}, p: {spearman_p:.4e}")
        
    
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
    

    
    # cortical
    to_remove = ''
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, FAD_df, reg_name='Isocortex', dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=9)
    plot_vals = plot_vals[['density_W', 'congo_density', 'acronym']].groupby(['acronym']).mean().reset_index()
    linear_regress(plot_vals, sav_fold, exp_name + '_CORTEX', figsize=(3.5, 3), xlim=(0, 18000), fontsize=12)
    
    # hippocampal
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, FAD_df, reg_name='Hippocampal formation', dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=11)
    plot_vals = plot_vals[['density_W', 'congo_density', 'acronym']].groupby(['acronym']).mean().reset_index()    
    linear_regress(plot_vals, sav_fold, exp_name + '_HIPPO', figsize=(3.5, 3), xlim=(0, 30000), fontsize=12)
    



    #%% Compare across different substructures if relationship to myelin holds
    to_remove = ''
    
    categories = ['Isocortex', 'Hippocampal formation', 'Cortical subplate', 'Thalamus', 'Hypothalamus', 'Midbrain', 'Cerebellum', 'fiber tracts']
    
    
    pool_plot = pd.DataFrame()
    for cat in categories:
        
        plot_vals, names_to_plot = get_subkeys_to_plot(df_means, FAD_df, reg_name=cat, dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=9)

        plot_vals['cat'] = cat
        
        pool_plot = pd.concat([pool_plot, plot_vals])
        
  
    pool_plot = pool_plot[['density_W', 'congo_density', 'acronym', 'cat']]
    pool_plot = pool_plot.groupby(['cat', 'acronym']).mean().reset_index()


    plt.figure(figsize=(3.5,3))
    fig = sns.scatterplot(data=pool_plot, x='density_W', y='congo_density', hue='cat')
    


    #%% Now compare CHANGE (i.e. loss --- in fold change) - relative to density of plaques
   
    AD_df = df_concat
    
    AD_df = AD_df[AD_df['exp'].isin(['5xFAD', '6mos'])]

    
    plot_vals = AD_df[['exp', 'acronym', 'density_W', 'st_level']]

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
    merged['density_diff'] = merged['density_W_5xFAD'] - merged['density_W_6mos']
    merged['abs_norm_diff'] = merged['density_diff']/merged['density_W_6mos']

    ### Filter by depth of structure
    # merged = merged[(merged['st_level'] > 5) & (merged['st_level'] < 9)]


    # linear_regress(merged, sav_fold, exp_name + '_HIPPO', x='abs_norm_diff', y='density_W', figsize=(3.5, 3), xlim=1, fontsize=12)
    
    # hippo_vals, hippo_names = get_subkeys_to_plot(df_means, FAD_df, reg_name='Hippocampal formation', dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=11)

    # hippo_acros = hippo_vals['acronym'].tolist()  # from earlier hippocampal plot
    # Filter for broad regions (st_level 5–9)
    lvl_low = 5
    lvl_high = 11
    
    filtered = merged[
        (merged['st_level'] >= lvl_low) & (merged['st_level'] <= lvl_high)
    ].copy()
    
    print(f"Regions before filtering: {len(merged)}")
    print(f"Regions after st_level filtering: {len(filtered)}")
    
    # Remove NaNs and infinities
    filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
    filtered.dropna(subset=['abs_norm_diff', 'density_W'], inplace=True)
    
    print(f"Regions after cleaning: {len(filtered)}")
    
    # Call your linear_regress function
    linear_regress(
        plot_vals_raw=filtered,
        sav_fold=sav_fold,
        exp_name=exp_name + '_BroadRegions',  # Adjust this to your preferred filename prefix
        x='abs_norm_diff',
        y='density_W',
        figsize=(3.5, 3),
        xlabel='Fold Change OL Density (5xFAD / Control)',
        ylabel='Plaque Density (5xFAD)',
        fontsize=12,
        ylim=(0, 5000),
        annotate_all=False,
        annotate_top_n=20  # Highlight top 20 plaque density regions
    )
    

#%% Compare to control

if exp_name == 'compare_control':
    AD_df = df_concat
    
    AD_df = AD_df[AD_df['exp'].isin(['5xFAD', '6mos'])]
    
    df_means = AD_df.groupby(AD_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
       
    
       
    
    ### Declare plotting variables
    palette = sns.color_palette("Set2")
    ### CAREFUL ---> df_means right now is pooled from ALL groups... for sorting...
        
    #%%% Volcano plot
   


   
    fontsize = 12
    exp_name = '5xFAD'
    df_pval = volcano_compare(AD_df, keys_df, compareby='exp', group1='6mos', group2='5xFAD',   ### group2 - group1 for means (so up is group2 > group1)
                    xplot='log2fold', 
                    
                    parcel_lvl_limit=9,   ### max (normally 9)
                    
                    thresh_pval=0.05, thresh_log2=0.3,
                    # thresh_pval=0.01, thresh_log2=0.3,  ### original threshold values
                    fontsize=fontsize, xlim=1, ylim=6, figsize=(3.2, 3))
    
    # estimate_cohen_d_for_log2fc(df_pval, threshold=0.3)
    # estimate_log2fc_for_target_d(df_pval, target_d=1.0)
    
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE.svg', format='svg', dpi=300)
    
    
    zzz
    #%%% Compare individual regions
    names_to_plot = ['Isocortex', 
                     # 'HPF', 
                     'HIP', 'RHP',
                     # 'CNU',
                     'STR', #'PAL',
                     'IB', #'TH', 'HY',   # Thalamus/Hypothalamus
                     'CBX', ### exclude cerebellar nuclei and arbor vitae later?
                     'fiber tracts', 
                     #'MB'
                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))
    
        
    plot_global(AD_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX',
                   ylim=25000, figsize=(7, 3))
    
    
    
    #%%% PLOT global regions of interest
    
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe
    
                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))
    
        
    plot_global(AD_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX',
                   ylim=25000, figsize=(7, 3))
    
    
    # plot raw numbers
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe
    
                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))
    
        
    plot_global(AD_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX_RAW_COUNT', dname='num_OLs_W',
                   ylim=600000, 
                   figsize=(7, 3),
                   logscale=True)
    
    
    # plt.ylim([0, 100000])
    # plt.tight_layout()
    
    
    names_to_plot = [
                     'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg',   ### Hippocampus
    
                        ### Retrohippocampal regions
                      #'RHP', 
                      'ENT', 'ENTl', 'ENTm', #'ENTmv',    ### Entorhinal areas divided into layer 1 - 6!!!
                      # 'ENTl1', 'ENTl2', 'ENTl3', 'ENTl5', 'ENTl6a',
                      # 'ENTm1', 'ENTm2', 'ENTm3', 'ENTm5', 'ENTm6',
                      
                        'PAR', #'PRE',  
                      
                      ### Para- post- and pre-subiculum (also divided into layer 1, 2, 3!!!)
                      
                      
                      ### Areas below here are too high to fit on ylim... need 40000
                        # 'POST',
                        'ProS', 
                        'SUB', #'SUBv', 'SUBd',   ### subiculum
                       # 'fornix system',
    
                     ]
    
    plot_global(AD_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_HIPPO',
                   ylim=25000, figsize=(4.2, 2.5))
    
    
    
    
        
        
    
        
    names_to_plot = [
                     'CBX', 'VERM', #vermal regions ### Cerebellum
    
                        ### Hemispheric regions, 
                      'HEM', 'SIM', 'AN', 'PRM', 'COPY', 'PFL', 'FL',    
                      
                      # 'FN', 'IP', 'DN',   ### Cerebellar nuclei
                       # 'arb',   ### arbor vitae
    
    
                     ]
    
    plot_global(AD_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CEREBELLUM',
                   ylim=25000, figsize=(4.2, 2.5))
    
    
    
    
    names_to_plot = [
                     'cbc', 'arb', #vermal regions ### Cerebellum
    
                        ### Hemispheric regions, 
                      'cc', 'cst', 
                      'tsp', 'rust', 
                      'mfbc', 'fxs', 'cing', 'act' 
                      'mfsbshy',    
                      
                      # 'FN', 'IP', 'DN',   ### Cerebellar nuclei
                       # 'arb',   ### arbor vitae
    
    
                     ]
    
    plot_global(AD_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_FIBER_TRACTS',
                   ylim=80000, figsize=(4.2, 2.5))
    
    
    zzz
    





# zzz


coords_df = pd.concat(all_coords_df)
coords_CONGO = pd.concat(all_coords_df_CONGO)


plt.figure(); plt.hist(coords_CONGO['vols'])


plt.figure();
sns.scatterplot(data=coords_CONGO, x='vols', y='mean_int')

    



all_total = []
dist_thresh = 50    
plt.figure();
for i, name in enumerate(fold_names):
    df_check = coords_df.loc[coords_df['exp_name'] == name]
    
    p2 = df_check[['dim0', 'dim1', 'dim2']] * 20   ### scale to microns
    
    p1 = coords_CONGO[['dim0', 'dim1', 'dim2']] * 20 
    p1 = np.array(p1)
    from scipy.spatial import cKDTree
    # Create a KDTree for SetB
    kdtree_b = cKDTree(np.array(p2))
    # For each point in SetA, find how many points in SetB are within the threshold distance
    neighbors_count = np.zeros(p1.shape[0], dtype=int)
    
    # Query the KDTree to get neighbors within the distance threshold
    for id_p, point in enumerate(p1):
        neighbors_count[id_p] = len(kdtree_b.query_ball_point(point, dist_thresh))
        
    print(i)
    plt.hist(neighbors_count)
    
    all_total.append(neighbors_count)
    
    coords_CONGO['NUM_NEIGHBORS_' + str(i)] = neighbors_count


plot_df = coords_CONGO[coords_CONGO['vols'] < 2000]
plot_df = plot_df[plot_df['vols'] > 500]

plot_df = plot_df[['NUM_NEIGHBORS_0', 'NUM_NEIGHBORS_1']]

plt.figure(); 
sns.boxplot(data=plot_df)







def make_atlas_mask(ref_atlas, cc_allen, cc_labs_allen, keys_df, region_names):
        
    all_sub_id = []
    ### just do striatum
    for name in region_names:
        sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name=name)
        all_sub_id.append(sub_idx)
    
    all_sub_id = [x for xs in all_sub_id for x in xs]
    
    sub_keys = keys_df.iloc[all_sub_id]
    sub_keys.reset_index(inplace=True, drop=True)
        
    sub_ids = np.asarray(sub_keys['ids'])

    remove_regions = np.zeros(np.shape(ref_atlas))
    for idx in sub_ids:
        cur_id = np.where(cc_labs_allen == idx)[0]
        
        #print(cur_id)
        if len(cur_id) == 0:  ### if it does not exists in atlas
            continue
        cur_coords = cc_allen[cur_id[0]]['coords']
        remove_regions[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
       
    
    iso_region = np.copy(ref_atlas)
    # iso_region[remove_regions > 0] = 0   ### delete all other regions
    iso_region[remove_regions == 0] = 0  ### keep current region
    
    return iso_region


reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'



ref_atlas = tiff.imread(reference_atlas)
ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)

cc_allen = regionprops(ref_atlas, cache=False)
cc_labs_allen = [region['label'] for region in cc_allen]
cc_labs_allen = np.asarray(cc_labs_allen)


### Grab just Isocortex

iso_region = make_atlas_mask(ref_atlas, cc_allen, cc_labs_allen, keys_df, region_names=['Isocortex'])

# import napari
# viewer = napari.Viewer()
# viewer.add_image(iso_region)
# viewer.show(block=True)


all_total = []
dist_thresh = 50    
plt.figure();
for i, name in enumerate(fold_names):
    df_check = coords_df.loc[coords_df['exp_name'] == name]


    p1 = coords_CONGO[['dim0', 'dim1', 'dim2']]
    p1 = np.asarray(np.round(p1), dtype=np.int32)
    # Apply clipping operation to each axis
    p1 = np.clip(p1, 0, np.array(iso_region.shape) - 1)  # subtract by 1
    
    ### mask out so only keep coords within isolated region
    p1_iso = p1[np.where(iso_region[p1[:, 0], p1[:, 1], p1[:, 2]])[0]]
    p1_iso *= 20   ### scale to microns
    
    
    
    ### also convert p2
    p2 = df_check[['dim0', 'dim1', 'dim2']] 
    p2 = np.asarray(np.round(p2), dtype=np.int32)
    # Apply clipping operation to each axis
    p2 = np.clip(p2, 0, np.array(iso_region.shape) - 1)  # subtract by 1
    
    ### mask out so only keep coords within isolated region
    p2_iso = p2[np.where(iso_region[p2[:, 0], p2[:, 1], p2[:, 2]])[0]]
    p2_iso *= 20   ### scale to microns   
    

    from scipy.spatial import cKDTree
    # Create a KDTree for SetB
    kdtree_b = cKDTree(np.array(p2_iso))
    # For each point in SetA, find how many points in SetB are within the threshold distance
    neighbors_count = np.zeros(p1_iso.shape[0], dtype=int)
    
    # Query the KDTree to get neighbors within the distance threshold
    for id_p, point in enumerate(p1_iso):
        neighbors_count[id_p] = len(kdtree_b.query_ball_point(point, dist_thresh))
        
    print(i)
    plt.hist(neighbors_count)
    
    all_total.append(neighbors_count)
    
    coords_CONGO['NUM_NEIGHBORS_' + str(i)] = neighbors_count


plot_df = coords_CONGO[coords_CONGO['vols'] < 2000]
plot_df = plot_df[plot_df['vols'] > 500]

plot_df = plot_df[['NUM_NEIGHBORS_0', 'NUM_NEIGHBORS_1']]

plt.figure(); 
sns.boxplot(data=plot_df)





### Then try splitting in white and grey matter
coords_df['location'] = 'WM'
cells_im, cell_pos = df_to_im(coords_df, atlas)
plot_max(cells_im)
    
loc_val = iso_layer[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]]

### show only GM cells
coords_df.loc[np.where(loc_val > 0)[0], 'location'] = 'GM'

large_GM = coords_df.iloc[np.where(coords_df['location'] == 'GM')[0]]

cells_im_LARGE_GM, cell_pos = df_to_im(large_GM, atlas)
plot_max(cells_im_LARGE_GM)

        
    
    



#%% EXTRACT FIMBRIA AND PLOT IT???

#%% Get coordinates of cells only in gray matter to plot
#sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='fornix system')

iso_region = make_atlas_mask(ref_atlas, cc_allen, cc_labs_allen, keys_df, region_names=['fimbria'])



### Now mask out and make meshes with these volumes and points

iso_cells = tmp_atlas[np.asarray(np.round(coords_df['Z_down']), dtype=np.int32),
                      np.asarray(np.round(coords_df['X_down']), dtype=np.int32),
                      np.asarray(np.round(coords_df['Y_down']), dtype=np.int32)]
iso_cells = np.where(iso_cells)[0]
iso_df = coords_df.iloc[iso_cells]

### Plot all cells here onto empty atlas
cells_im = np.zeros(np.shape(atlas))
cells_im[np.asarray(np.round(iso_df['Z_down']), dtype=np.int32),
         np.asarray(np.round(iso_df['X_down']), dtype=np.int32),
         np.asarray(np.round(iso_df['Y_down']), dtype=np.int32)] = 255



plot_max(myelin)

import napari
viewer = napari.Viewer()
#napari.run()

# add image
viewer.add_image(myelin)
viewer.add_image(tmp_atlas)
viewer.add_image(cells_im)
viewer.add_image(atlas)



#%% Do clustering
XY_res = 1.152035240378141
Z_res = 5

from sklearn.preprocessing import StandardScaler

data = np.asarray([iso_df['Z_scaled'].values, iso_df['X_scaled'].values, iso_df['Y_scaled'].values])
data = np.transpose(data)

### SCALE Z dimension!!!
data[:, 0] = data[:, 0] * Z_res
data[:, 1] = data[:, 1] * XY_res
data[:, 2] = data[:, 2] * XY_res


### only get one hemisphere
data = data[np.where(data[:, 2] > 6400)[0]]



X = data
#X = StandardScaler().fit_transform(data)

from sklearn.cluster import DBSCAN, HDBSCAN

db = DBSCAN(eps=15, min_samples=5).fit(X)

#db = HDBSCAN().fit(X)

labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(n_clusters_)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=db.labels_, cmap='hsv')


clusters = X[np.where(db.labels_ > 0)]
clust_colors =  db.labels_[np.where(db.labels_ > 0)]
ax.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], c=clust_colors, cmap='hsv', s=1)



### plot 2D
fig = plt.figure()
ax = fig.add_subplot()
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=db.labels_, cmap='hsv')


clusters = X[np.where(db.labels_ > 0)]
clust_colors =  db.labels_[np.where(db.labels_ > 0)]
ax.scatter(clusters[:, 1], clusters[:, 2], c=clust_colors, cmap='hsv', s=1)

