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

list_brains = get_metadata(mouse_num = 'all')
# list_brains = get_metadata(mouse_num = ['M246'])
if new_large_OL:
    list_brains = get_metadata(mouse_num = ['M260', 'M286'])

# print('CUPRIZONE AND OLD BRAIN (anything with lots of lipofuscin) currently using -10grid (only 1st old brain, all newer analyzed old brains using -15grid)')

sav_fold = '/media/user/8TB_HDD/Plot_outputs/'


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


#%%%% Parse pickles from all folders
""" Loop through all the folders and pool the dataframes!!!"""

# all_coords_df = []
all_keys_df = []
for id_f, info in enumerate(list_brains):

    fold = info['path'] + info['name'] + '_postprocess'

    pkl_to_use = info['pkl_to_use']

    if not ANTS:
        keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid.pkl'))    
    else:
        # keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY.pkl'))   
        
        if pkl_to_use == 'MYELIN':
            keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl'))   
            
 
        elif pkl_to_use == 'CUBIC':
            keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE_CUBIC.pkl')) 
            
            
    try:    
        keys_df = pd.read_pickle(keys[0])
    except:
        print('Missing: ' + fold)
        continue
    # coords_df = pd.read_pickle(coords[0])
    
    # print('number of cells: ' + str(len(coords_df)))
    
    keys_df['acronym'] = keys_tmp['acronym']
    keys_df['dataset'] = info['num']
    keys_df['exp'] = info['exp']
    keys_df['sex'] = info['sex']
    keys_df['age'] = info['age']   
    
    
    
    drop_ids = []
    #%%% REMOVE ADDITIONAL TORN TISSUE REGIONS - as specified in metadata (including replacements if given hemisphere)
    
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



    # all_coords_df.append(coords_df)
    all_keys_df.append(keys_df)


df_concat = pd.concat(all_keys_df)


# zzz


#%%% Experiment name


if new_large_OL:
    exp_name = 'LARGE'
    
else:
    exp_name = 'Cuprizone'
    exp_name = 'Strain diff'
    exp_name = 'pooled P60'
    exp_name = 'sex_diff'
    exp_name = 'Aging'

plot_all = False


fontsize = 14

fig_stand = np.asarray([4.5, 5])
fig_long = np.asarray([4.8,4])

P60_x_lim = 18000


zzz


#%% Plot figure 1 ASPA counts

data = [
        {'Group':'P60', 'Area':'GM', 'Percent':100},
        {'Group':'P60', 'Area':'GM', 'Percent':100},
        {'Group':'P60', 'Area':'GM', 'Percent':100},
        {'Group':'P60', 'Area':'GM', 'Percent':100},
        {'Group':'P60', 'Area':'WM', 'Percent':100},
        {'Group':'P60', 'Area':'WM', 'Percent':100},
        {'Group':'P60', 'Area':'WM', 'Percent':100},
        {'Group':'P60', 'Area':'WM', 'Percent':100},
        
        {'Group':'FVB', 'Area':'GM', 'Percent':100},
        {'Group':'FVB', 'Area':'GM', 'Percent':100},
        {'Group':'FVB', 'Area':'GM', 'Percent':100},
        {'Group':'FVB', 'Area':'GM', 'Percent':100},
        {'Group':'FVB', 'Area':'WM', 'Percent':97},
        {'Group':'FVB', 'Area':'WM', 'Percent':100},
        {'Group':'FVB', 'Area':'WM', 'Percent':100},
        {'Group':'FVB', 'Area':'WM', 'Percent':97},

        {'Group':'P620', 'Area':'GM', 'Percent':100},
        {'Group':'P620', 'Area':'GM', 'Percent':100},
        {'Group':'P620', 'Area':'GM', 'Percent':100},
        {'Group':'P620', 'Area':'GM', 'Percent':100},
        {'Group':'P620', 'Area':'WM', 'Percent':100},
        {'Group':'P620', 'Area':'WM', 'Percent':100},
        {'Group':'P620', 'Area':'WM', 'Percent':100},
        {'Group':'P620', 'Area':'WM', 'Percent':100},        

        ]

aspa_counts = pd.DataFrame(data)

palette = sns.color_palette("Set2")
palette[2] = palette[3]    

plt.figure(figsize=(3,2.5))
ax = sns.barplot(
    data=aspa_counts, #kind="bar",
    x="Area", y="Percent", hue="Group",
    errorbar="sd", palette=palette,  #  alpha=.6, height=6,
    # color='k'
)
# plt.setp(ax.patches, linewidth=5)

for patch in ax.patches:
    # clr = patch.get_facecolor()
    patch.set_edgecolor('k')


sns.stripplot(
    data=aspa_counts, #kind="bar",
    x="Area", y="Percent", hue="Group",
    color='k',
    dodge=True, alpha=0.4, ax=ax, jitter=0.25
)



# g.despine(left=True)
ax.set_xlabel("")
ax.set_ylabel("% ASPA+;EGFP+/EGFP+", fontsize=fontsize)
# ax.legend.set_title("")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.yticks(fontsize=fontsize - 2)
plt.xticks(fontsize=fontsize - 2)
# plt.ylim([0, 100])

plt.tight_layout()

plt.savefig(sav_fold + 'ASPA_counts.png', format='png', dpi=300)
plt.savefig(sav_fold + 'ASPA_counts.svg', format='svg', dpi=300)



#%% CUPRIZONE pooling
if exp_name == 'Cuprizone' or plot_all:
    exp_name = 'Cuprizone'
    cup_df = df_concat[df_concat['exp'].isin(['P60', 'Cuprizone', 'Recovery', 'Cuprizone_NEW', 'Recovery_NEW'])]

    df_means = cup_df.groupby(cup_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']    

    ### Declare plotting variables
    palette = sns.color_palette("Set2")
    

    
    #%%% Plot by layer
    cup_only_df = df_concat[df_concat['exp'].isin(['Cuprizone'])]
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
  
    plot_by_layer(cup_only_df, [], names_to_plot, sav_fold, exp_name, plot_name='CUPRIZONE', figsize=(3.2, 3))

        
    #%%% Plot by layer RECOVERY
    rec_only_df = df_concat[df_concat['exp'].isin(['Recovery'])]
    plot_by_layer(rec_only_df, [], names_to_plot, sav_fold, exp_name, plot_name='RECOVERY', figsize=(3.2, 3))

    #%%% Plot by layer the absolute DIFFERENCE

    P60_df = df_concat[df_concat['exp'].isin(['P60'])]
    plot_by_layer(cup_only_df, P60_df, names_to_plot, sav_fold, exp_name, plot_name='NORM_cup', figsize=(3.2, 3),
                  ylim=[0, 1.2])


    plot_by_layer(rec_only_df, P60_df, names_to_plot, sav_fold, exp_name, plot_name='NORM_recov', figsize=(3.2, 3),
                  ylim=[0, 1.2])


    #%%% Cuprizone RECOVERY
    ### First plot curve of each layer normalized to starting P60 density
    
    ### Include expected density at P120 at the very end
    
    # names_to_plot = ['SSp', 'MOp','RSP', 'SSs', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'TEa']
    
    # names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    
    
    names_to_plot = ['SSp', 'MOp','RSP', 'SSs', 'MOs', 'VIS', 'AUD', 'PERI', 'TEa', 'PL', 'ACA', 'AI',
                     'ORB', 'TEa', 'ILA']
    
    
    palette = sns.color_palette("Set2", len(names_to_plot))
    #styles = ['-', '--', '-.', ':']
    styles=['-']
    
    layers = ['1', '2/3', '4', '5', '6']
    
    fontsize = 14
    
    #%%% Compare layers for each cortical region
    # all_layers = []

    fig1, ax1 = plt.subplots(5, 3, figsize=(8, 10), sharex=True, sharey=True)
    fig2, ax2 = plt.subplots(5, 3, figsize=(8, 10), sharex=True, sharey=True)
    for i_x, name in enumerate(names_to_plot):

        for i_n, layer in enumerate(layers):
            # if name matches the first half at minimum, then go and plot
            match = cup_df[cup_df['acronym'].str.contains(name) == True]
            if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
                match = match[match['acronym'].str.contains('RSPd4') == False]   
            match=match.reset_index()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                           
                                                                                                                                                                                                    
            ############## HAVE TO SUM ALL AREAS FIRST AND THEN RE-CALCULATE DENSITY so it's ACROSS the layer, 
            ##### rather than an AVERAGE of all areas
            lay_df = match[match['names'].str.contains(layer) == True]
            lay_df = lay_df.groupby(['dataset', 'age']).sum()
            lay_df['density_W'] = lay_df['num_OLs_W']/lay_df['atlas_vol_W']
            
            lay_df = lay_df.reset_index()
        
        
            ##############
        
        
            if len(lay_df) == 0:
                continue
        
                        
            cur_ax = sns.lineplot(ax=ax1.flatten()[i_x], x=lay_df['age'], y=lay_df['density_W'], label=layer, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                         #order=['P60', 'P120', 'P240', 'P620'], 
                         errorbar='se',
                         marker='o',
                         markersize=8, linewidth=2).set(title=name) 


            lay_df['density_NORM'] = lay_df['density_W']/np.nanmean(lay_df.iloc[np.where(lay_df['age'] == 60)[0]]['density_W'])
            
            
            sns.lineplot(ax=ax2.flatten()[i_x], x=lay_df['age'], y=lay_df['density_NORM'], label=layer, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                         errorbar='se',
                         marker='o',
                         markersize=8, linewidth=2).set(title=name) 
            ax2.flatten()[i_x].set_ylim(0, 1.5)
            
            
            

            
            
    # ### Remove legends
    # for ax in ax1.flatten():
    #     ax.legend([],[], frameon=False)

    # for ax in ax2.flatten():
    #     ax.legend([],[], frameon=False)

    ### Remove legends
    for ax in ax1.flatten():
        ax.legend([],[], frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        ax.set_ylim([0, 30000])
        ax.ticklabel_format(axis='y', scilimits=(-4, 4))
        ax.xaxis.get_offset_text().set_fontsize(fontsize-2)
        
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        
    for ax in ax2.flatten():
        ax.legend([],[], frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('')
        ax.set_ylabel('')

        
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        
        
        
        
    handles, labels = ax1[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right')
    fig2.legend(handles, labels, loc='upper right')



    fig1.text(0.04, 0.5, 'Density (cells/mm\u00b3)', va='center', rotation='vertical', fontsize=fontsize)
    fig2.text(0.04, 0.5, 'Normalized cell density', va='center', rotation='vertical', fontsize=fontsize)


    plt.figure(fig1)
    plt.savefig(sav_fold + exp_name +'_by_layer_and_region_CUPRIZONE.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_by_layer_and_region_CUPRIZONE.svg', format='svg', dpi=300)
    

    plt.figure(fig2)
    plt.savefig(sav_fold + exp_name +'_by_layer_and_region_CUPRIZONE_NORM.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_by_layer_and_region_CUPRIZONE_NORM.svg', format='svg', dpi=300)
    

    
    #%%% Plot white matter and other sensory areas

    # to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|VISpl|'
    # plot_vals = plot_stripes_from_df(df_means, cup_df, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Thalamus', dname='density_W',
    #             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=8, leg_loc='lower right',
    #             x_lab='Density of OLs (cells/mm\u00b3)', y_lab='', xlim=[0, 60000], name='medium_level_CUPRIZONE') 

    
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
    palette = sns.color_palette("husl", len(names_to_plot))
    
    palette[-1] =sns.color_palette("Set2", len(names_to_plot))[5]
    
    compare_GLOBAL_regions(cup_df, names_to_plot, palette, xlim=[58, 120], ylim=50000, ylim_norm=[0,1.2], sav_name='AGING', sav_fold=sav_fold, exp_name=exp_name, 
                           fontsize=14, add_zero=False, figsize=(4,3))


    
    # Compare fiber tracts P60 vs. P640
    for name in names_to_plot:
        match = cup_df[cup_df['acronym'].str.fullmatch(name) == True]
        
        P60 = match[match['exp'] == 'P60']
        Cup = match[match['exp'] == 'Cuprizone']    
        print('Change in density: ' + str(np.mean(Cup['density_W']) - np.mean(P60['density_W'])) + name)
    
    # also compare fold change
    for name in names_to_plot:
        match = cup_df[cup_df['acronym'].str.fullmatch(name) == True]
        
        P60 = match[match['exp'] == 'P60']
        Cup = match[match['exp'] == 'Cuprizone']     
        print('Change in density: ' + str((np.mean(Cup['density_W']) - np.mean(P60['density_W']))/np.mean(P60['density_W'])) + name)
    
                
    
    
    ### Compare pvalues for RSP vs. SSp after injury and during recovery
    match = cup_df[cup_df['acronym'].str.fullmatch('RSP') == True]
    Cup_RSP = match[match['exp'] == 'Cuprizone']
    Rec_RSP = match[match['exp'] == 'Recovery']
    print(stats.sem(Cup_RSP['density_W']))   
    # print(stats.sem(Rec_RSP['density_W']))   
    
    match = cup_df[cup_df['acronym'].str.fullmatch('SSp') == True]
    Cup_SSp = match[match['exp'] == 'Cuprizone']
    Rec_SSp = match[match['exp'] == 'Recovery']
    print(stats.sem(Cup_SSp['density_W']))   
    # print(stats.sem(Rec_SSp['density_W']))      
    
    
    tstat, p = stats.ttest_ind(Cup_RSP['density_W'], Cup_SSp['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('Cup_RSP:' + str(np.mean(Cup_RSP['density_W'])) + ' Cup_SSp: ' + str(np.mean(Cup_SSp['density_W'])))

    
    tstat, p = stats.ttest_ind(Rec_RSP['density_W']/np.mean(Cup_RSP['density_W']), Rec_SSp['density_W']/np.mean(Cup_SSp['density_W']), equal_var=True, alternative='two-sided')
    print(p)
    print('Cup_RSP:' + str(np.mean(Rec_RSP['density_W']/np.mean(Cup_RSP['density_W']))) + ' Cup_SSp: ' + str(np.mean(Rec_SSp['density_W']/np.mean(Cup_SSp['density_W']))))

    print(stats.sem(Rec_RSP['density_W']/np.mean(Cup_RSP['density_W'])))
    print(stats.sem(Rec_SSp['density_W']/np.mean(Cup_SSp['density_W'])))
    
    
    
    ### Then compare L4 vs. L5
    regions_compare = ['SSp', 'SSs', 'VISp', 'AUDp']
    for reg in regions_compare:
        match = cup_df[cup_df['acronym'].str.contains(reg) == True]
        match = match[match['exp'] == 'Cuprizone']
        # all_df = match.iloc[np.where((match['names'].str.contains('2/3') == True) | (match['names'].str.contains('4') == True))[0]]

        all_df = match.iloc[np.where((match['names'].str.contains('4') == True))[0]]
        sum_df = all_df.groupby(['dataset']).sum()
    
        sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
        sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
        sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
    
    
        all_df = match.iloc[np.where((match['names'].str.contains('5') == True))[0]]
        sum_df_5 = all_df.groupby(['dataset']).sum()
    
        sum_df_5['density_L'] = sum_df_5['num_OLs_L']/sum_df_5['atlas_vol_L']
        sum_df_5['density_R'] = sum_df_5['num_OLs_R']/sum_df_5['atlas_vol_R']
        sum_df_5['density_W'] = sum_df_5['num_OLs_W']/sum_df_5['atlas_vol_W']
        
    
        tstat, p = stats.ttest_ind(sum_df_5['density_W'], sum_df['density_W'], equal_var=True, alternative='two-sided')
        print(p)
        print('L5:' + str(np.mean(sum_df_5['density_W'])) + ' L4' + reg + str(np.mean(sum_df['density_W'])))
        print(stats.sem(sum_df_5['density_W']))   
        print(stats.sem(sum_df['density_W']))   
      
    
    

    
    
    
    #%%% PLOT global regions of interest

    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX_CUPRIZONE',
                   ylim=15000, figsize=(7, 3))
    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX_NORM_CUPRIZONE',
                   ylim=1.4, figsize=(7, 3), norm_name='P60')
    
    names_to_plot = [
                     'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg',   ### Hippocampus

                        ### Retrohippocampal regions
                      #'RHP', 
                      'ENT', 'ENTl', 'ENTm', #'ENTmv',    ### Entorhinal areas divided into layer 1 - 6!!!
                      # 'ENTl1', 'ENTl2', 'ENTl3', 'ENTl5', 'ENTl6a',
                      # 'ENTm1', 'ENTm2', 'ENTm3', 'ENTm5', 'ENTm6',
                      
                      # 'PAR', 'PRE',  
                      
                      ### Para- post- and pre-subiculum (also divided into layer 1, 2, 3!!!)
                      
                      
                      ### Areas below here are too high to fit on ylim... need 40000
                      # 'POST',
                      # 'SUB', #'SUBv', 'SUBd'   ### subiculum

                     ]

    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_HIPPO_CUPRIZONE',
                   ylim=15000, figsize=(4, 3.5))
    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_HIPPO_NORM_CUPRIZONE',
                   ylim=1.4, figsize=(4, 3.5), norm_name='P60')
    
    
    
        
    names_to_plot = [
                     'CBX', 'VERM', #vermal regions ### Cerebellum

                        ### Hemispheric regions, 
                      'HEM', 'SIM', 'AN',# 'PRM', 'COPY', 'PFL', 'FL',    
                      
                       # 'FN', 'IP', 'DN',   ### Cerebellar nuclei
                      # 'arb',   ### arbor vitae

                     ]

    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CEREBELLUM_CUPRIZONE',
                   ylim=12000, figsize=(2.5, 3.5))
    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CEREBELLUM_NORM_CUPRIZONE',
                   ylim=1.4, figsize=(2.5, 3.5), norm_name='P60')
    
    
    names_to_plot = [
                     # 'CBX', 'VERM', #vermal regions ### Cerebellum

                        ### Hemispheric regions, 
                      # 'HEM', 'SIM', 'AN',# 'PRM', 'COPY', 'PFL', 'FL',    
                      
                        'FN', 'IP', 'DN',   ### Cerebellar nuclei
                       'arb',   ### arbor vitae

                     ]

    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CEREBELLAR_NUCLEI_CUPRIZONE',
                   ylim=80000, figsize=(2.3, 3.5))
    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CEREBELLAR_NUCLEI_NORM_CUPRIZONE',
                   ylim=1.4, figsize=(2.3, 3.5), norm_name='P60')
    
    
    
    #%% Thalamus
    names_to_plot = [
                    'DORsm', 
                    'VENT', 'VAL', 'VM',  'VP',      ### Ventral thalamus
                    'GENd', #'MG', 'MGd', 'MGv', 'MGm', 
                    'LGd',
                    'DORpm',
                    # 'LAT', 'ANT', 'MED', 'MTN', 'ILM', 'RT',   ### Lateral, anterior, medial, midline, intralaminar, reticular groups (nuclei mostly) 
                    'GENv',               ### Ventral thalamus
                    'IGL', 'LGv',                            # Lateral geniculate nucleus

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_THALAMUS_CUPRIZONE',
                   ylim=60000, figsize=(4, 3.5))
    
    plot_global(cup_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_THALAMUS_NORM_CUPRIZONE',
                   ylim=1.4, figsize=(4, 3.5), norm_name='P60')
    
    
    
    
    ### stats for comparison
    # Compare VP and VM areas
    match = cup_df[cup_df['acronym'].str.fullmatch('VP') == True]
    VP = match[match['exp'] == 'Recovery']
    match = cup_df[cup_df['acronym'].str.fullmatch('VM') == True]
    VM = match[match['exp'] == 'Recovery']    

    print(stats.sem(VP['density_W']))   
    print(stats.sem(VM['density_W']))   

    tstat, p = stats.ttest_ind(VP['density_W'], VM['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('VP:' + str(np.mean(VP['density_W'])) + ' VM: ' + str(np.mean(VM['density_W'])))


    
    
    
    
    
#%% STRAIN pooling

if exp_name == 'Strain diff' or plot_all:
    exp_name = 'Strain diff'
    
    
    strain_df = df_concat[df_concat['exp'].isin(['P60', 'CD1', 'CD1_NEW', 'FVB', 'FVB_NEW', 'P60_NEW', 'P60_126'])]
    df_means = strain_df.groupby(strain_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']


    ### Declare plotting variables
    palette = sns.color_palette("Set2")
    ### CAREFUL ---> df_means right now is pooled from ALL groups... for sorting...

    #%%% Volcano plot
    volcano_compare(strain_df, keys_df, compareby='exp', group1='P60', group2='FVB',
                    xplot='log2fold', thresh_pval=0.01, thresh_log2=0.3,
                    fontsize=fontsize, xlim=1, ylim=6, figsize=(3.2, 3))
    
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE.svg', format='svg', dpi=300)
    
    
    
    
    #%%% PLOT global regions of interest
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(strain_df, names_to_plot, [palette[0], palette[3]], sav_fold, exp_name + '_GLOBAL_COMPARISON_straindiff',
                   ylim=20000, figsize=(6, 3), fontsize=14)
    
        
    
    
    
    #%% Compare individual regions and get pvalues
    
    # Compare RSP
    match = strain_df[strain_df['acronym'].str.fullmatch('RSP') == True]
    B6 = match[match['exp'] == 'P60']
    FVB = match[match['exp'] == 'FVB']
    # motor_df = match[match['names'].str.contains('2/3') == True]
    print(stats.sem(B6['density_W']))   
    print(stats.sem(FVB['density_W']))   
    

    tstat, p = stats.ttest_ind(B6['density_W'], FVB['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('B6:' + str(np.mean(B6['density_W'])) + ' FVB: ' + str(np.mean(FVB['density_W'])))

    
    # Compare Visual
    match = strain_df[strain_df['acronym'].str.fullmatch('VISp') == True]
    B6 = match[match['exp'] == 'P60']
    FVB = match[match['exp'] == 'FVB']
    # motor_df = match[match['names'].str.contains('2/3') == True]
    print(stats.sem(B6['density_W']))   
    print(stats.sem(FVB['density_W']))   
    

    tstat, p = stats.ttest_ind(B6['density_W'], FVB['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('B6:' + str(np.mean(B6['density_W'])) + ' FVB: ' + str(np.mean(FVB['density_W'])))

    
    # Compare Auditory
    match = strain_df[strain_df['acronym'].str.fullmatch('AUDp') == True]
    B6 = match[match['exp'] == 'P60']
    FVB = match[match['exp'] == 'FVB']
    # motor_df = match[match['names'].str.contains('2/3') == True]
    print(stats.sem(B6['density_W']))   
    print(stats.sem(FVB['density_W']))   
    

    tstat, p = stats.ttest_ind(B6['density_W'], FVB['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('B6:' + str(np.mean(B6['density_W'])) + ' FVB: ' + str(np.mean(FVB['density_W'])))





#%% SEX pooling
if exp_name == 'sex_diff' or plot_all:
    exp_name = 'sex_diff'
    fontsize=14

    ### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
    sex_df = df_concat[df_concat['exp'].isin(['P60', 'P60_F', 'P60_nosunflow'])]
    
    sex_df.loc[sex_df['sex']== 'F', 'exp'] = 'Female'
    sex_df.loc[sex_df['sex']== 'M', 'exp'] = 'Male'
    
    
    df_means = sex_df.groupby(sex_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
    

    ### Declare plotting variables
    palette = sns.color_palette("Set2")
    ### CAREFUL ---> df_means right now is pooled from ALL groups... for sorting...
    
    #%%% Volcano plot
    volcano_compare(sex_df, keys_df, compareby='sex', group1='M', group2='F',
                    xplot='log2fold', thresh_pval=0.01, thresh_log2=0.3,
                    fontsize=fontsize, xlim=1, ylim=6, figsize=(3.2, 3))
    
    
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_VOLCANO_COMPARE.svg', format='svg', dpi=300)
    
    
    

    #%%% PLOT global regions of interest
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(sex_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_sexdiff',
                   ylim=18000, figsize=(6, 3), fontsize=14)
    
    
    
    #%% Compare individual regions and get pvalues
    
    # Compare Cortical Amygdalar area
    match = sex_df[sex_df['acronym'].str.fullmatch('COA') == True]
    male = match[match['sex'] == 'M']
    female = match[match['sex'] == 'F']
    # motor_df = match[match['names'].str.contains('2/3') == True]
    print(stats.sem(male['density_W']))   
    print(stats.sem(female['density_W']))   
    

    tstat, p = stats.ttest_ind(male['density_W'], female['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('Male:' + str(np.mean(male['density_W'])) + ' Female: ' + str(np.mean(female['density_W'])))

    
    # Compare Hypothalamus
    match = sex_df[sex_df['acronym'].str.fullmatch('HY') == True]
    male = match[match['sex'] == 'M']
    female = match[match['sex'] == 'F']
    # motor_df = match[match['names'].str.contains('2/3') == True]
    print(stats.sem(male['density_W']))   
    print(stats.sem(female['density_W']))   
    

    tstat, p = stats.ttest_ind(male['density_W'], female['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('Male:' + str(np.mean(male['density_W'])) + ' Female: ' + str(np.mean(female['density_W'])))
    
    
    
    # def get_pvalue(df1, df2, value='density_W'):
        
        
        
    #%% Compare weights of males and females
    M = [20, 19, 25.2, 25.5]
    F = [20, 16, 17.5, 18]
    
    mean = np.mean(M)
    sem = stats.sem(M)
    print('Male: ' + str(mean) + ' ' + str(sem))
    
    mean = np.mean(F)
    sem = stats.sem(F)
    print('Female: ' + str(mean) + ' ' + str(sem))
        
    tstat, p = stats.ttest_ind(M, F, equal_var=True, alternative='two-sided')
    print(p)    
	

    
        



#%% P60 pooling
if exp_name == 'pooled P60' or plot_all:

    exp_name = 'pooled P60'
    
    ### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
    pool_all_df = df_concat[df_concat['exp'].isin(['P60', 'P60_F'])]
    df_means = pool_all_df.groupby(pool_all_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
    
    

    #%%% Variance plot across all layers sorted by cortex, midbrain, ect...
    
    ## MAKE SURE TO TURN OFF DROP_ID FOR THIS!!! WANT TO KEEP REGIONS WITH HIGH VARIANCE TO SHOW IT
    
    areas_to_graph = ['Isocortex', 'Hippocampal formation', 'Hypothalamus', 'Thalamus', 'Midbrain', 'Cerebellum']
    # to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|VISpl|'
    to_remove_CORTEX = ''
    #palette = sns.color_palette("husl")
    palette = sns.color_palette("Set2")
    plt.figure(figsize=(3.6, 3))
    all_areas = []
    for area in areas_to_graph:
        plot_vals, names_to_plot = get_subkeys_to_plot(df_means, pool_all_df, reg_name=area, dname='density_W', 
                                                       to_remove=to_remove_CORTEX, lvl_low=5, lvl_high=9)
        mean = plot_vals.groupby(['acronym', 'names']).mean(numeric_only=True).reset_index()
        std = plot_vals.groupby(['acronym', 'names']).std(numeric_only=True).reset_index()
        
        mean['std'] = std['density_W']
        mean['mean'] = mean['density_W']
        mean['cv'] = std['density_W']/mean['density_W']   ### also calculate normalized std (coefficient of variation)
    
        ### combine mean and variance into same dataframe
        combined = mean.sort_values(by='cv', ascending=False).reset_index()
        
        # sns.barplot(std, y=std.index, x='density_W', order=std['acronym'], color='grey',
        #             errorbar=None)
        combined['Parcellation'] = area
        
        all_areas.append(combined)
        
    all_areas = pd.concat(all_areas).reset_index()
    
    # get means
    print(all_areas.groupby('Parcellation')['cv'].mean())
    print(all_areas.groupby('Parcellation')['cv'].sem())
        
    ax = sns.barplot(x=all_areas["cv"], y=all_areas["acronym"], orient="h", hue=all_areas['Parcellation'],
                     palette=palette)
    

    outlier_ids = np.where(all_areas["cv"] > 0.3)[0]
    outliers = all_areas.iloc[outlier_ids]
    
    all_texts = []
    for idx, row in outliers.iterrows():
        all_texts.append(plt.text(row['cv'] + 0.01, idx, 
                  row['acronym'], ha='center', va='center',
                          size=10, color='black', weight='normal'))


    sns.move_legend(ax, loc='lower right', frameon=False, title='', fontsize=fontsize)
    plt.yticks(fontsize=fontsize - 2)
    plt.yticks([])
    plt.xticks(fontsize=fontsize - 2)
    plt.xlim([0, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel('Brain regions', fontsize=fontsize)
    plt.xlabel('Coefficient of variation', fontsize=fontsize)
    plt.tight_layout()

    adjust_text(all_texts, objects=ax.containers[1], only_move='x+',
                arrowprops=dict(arrowstyle='->', color='red'))
    


    plt.savefig(sav_fold + exp_name +'_all_brain_regions_VARIABILITY.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_all_brain_regions_VARIABILITY.svg', format='svg', dpi=300)
    
    
    
    

    #%%% Compare across hemispheres
    # pooled_df = df_concat[df_concat['dataset'].isin(['M229', 'M115', 'M223', 'M126'])]
    
    # pooled_df = df_concat
    
    pooled_df = pool_all_df
    
    df_means = pooled_df.groupby(pooled_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
        
    mean_hemisphere = pooled_df.groupby(['acronym', 'names']).mean(numeric_only=True)
    
    plt.figure(figsize=(3.5, 3.5))
    ax = plt.gca()
    pl = sns.regplot(x='density_R', y='density_L', data=mean_hemisphere.dropna(), scatter_kws={"color": "grey", 's':10}, line_kws={"color": "red", "alpha":0.2}, ax=ax)
    
    #calculate slope and intercept of regression equation
    slope, intercept, r, p, sterr = stats.linregress(x=pl.get_lines()[0].get_xdata(),
                                                           y=pl.get_lines()[0].get_ydata())
    
    #display slope and intercept of regression equation
    print(slope)


    r,p = stats.pearsonr(mean_hemisphere.dropna()['density_R'], mean_hemisphere.dropna()['density_L'])
    print(r)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)

    plt.xlim([0, 80000])
    plt.ylim([0, 80000])
    ax.ticklabel_format(axis='x', scilimits=[-3, 3])  ### set to be order of magnitude
    ax.ticklabel_format(axis='y', scilimits=[-3, 3])
    
    
    ### Find extreme outliers
    # check_diff = (np.abs(mean_hemisphere['density_L'] - mean_hemisphere['density_R']))/((np.abs(mean_hemisphere['density_L'] + mean_hemisphere['density_R'])/2))
    # outlier_idx = np.where(check_diff > 0.4)[0]
    # outliers = mean_hemisphere.iloc[outlier_idx].reset_index()
    
    # all_texts = []
    # for idx, row in outliers.iterrows():
    #     all_texts.append(plt.text(row['density_L'] +0.01, row['density_R'], 
    #              row['acronym'], ha='center', va='center',
    #                       size=10, color='black', weight='normal'))
    # adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='red'))


    ### Force calculate p value
    from mpmath import mp
    # mp.dps = 1000
    
    r = mp.mpf(r)
    n = len(mean_hemisphere.dropna())
    
    x = (-abs(r) + 1)/2  # shift per `loc=-1`, scale per `scale=2`
    p = 2*mp.betainc(n/2 - 1, n/2 - 1, 0, x, regularized=True)
    print(p)

        
    plt.xlabel('Density right (cells/mm\u00b3)', fontsize=fontsize)
    plt.ylabel('Density left (cells/mm\u00b3)', fontsize=fontsize)
    plt.tight_layout()
    
    
    
    
    plt.savefig(sav_fold + exp_name + '_LvsR_correlation.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_LvsR_correlation.svg', format='svg', dpi=300)

    ### Find regions that do NOT match
    # mean_hemisphere['num_OLs_absdiff'] = abs(mean_hemisphere['num_OLs_L'] - mean_hemisphere['num_OLs_R'])
    # mean_hemisphere['num_OLs_scaleddiff'] = mean_hemisphere['num_OLs_absdiff']/mean_hemisphere['num_OLs_W']   
    
    
    
    
    
    
    
    #%%% Plot by layer
    plt.figure(figsize=(3.6, 3))
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    #names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = pooled_df[pooled_df['acronym'].str.contains(name) == True]
        if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
            match = match[match['acronym'].str.contains('RSPd4') == False]   
        
        layers = ['1', '2/3', '4', '5', '6']
        
        all_layers = []
        for layer in layers:
        
            lay_df = match[match['names'].str.contains(layer) == True]
        
            if len(lay_df) == 0:
                continue
        
            sum_df = lay_df.groupby(['dataset']).sum()
        
            sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
            sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
        
            sum_df['layer'] = layer
            
            all_layers.append(sum_df)
            
        df_layers = pd.concat(all_layers)
            
        ### SKIP if wasnt subdivided into smaller layer units
        if df_layers['density_W'].isna().any():
            continue
            
        # plt.figure()
        # sns.boxplot(x=df_layers['layer'], y=df_layers['density_W'])
        sns.lineplot(x=df_layers['layer'], y=df_layers['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                     errorbar=('se'))
    
    
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    plt.legend(loc = 'lower right', frameon=False)
    
    
    plt.xlabel('Cortical layer', fontsize=fontsize)
    plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    
    
    ax.set_yscale('log')
    plt.ylim([0, 100000])
    plt.tight_layout()
    
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG.svg', format='svg', dpi=300)


    # ### Then plot it linear
    plt.legend(loc = 'upper left', frameon=False)
    ax.set_yscale('linear')
    plt.ylim([0, 30000])  ### this ruins log plot...
    ax = plt.gca()
    
        
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    
    
    
    plt.tight_layout()
    plt.savefig(sav_fold + exp_name + '_by_LAYERS.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_by_LAYERS.svg', format='svg', dpi=300)




    #%% Compare individual layers and regions and get pvalues
    
    # First get motor cortex layer 2/3
    match = pooled_df[pooled_df['acronym'].str.contains('MOp') == True]
    motor_df = match[match['names'].str.contains('2/3') == True]
    print(stats.sem(motor_df['density_W']))   
    
    
    # Then get other cortices but with layers 2/3 + 4 summed
    regions_compare = ['SSp', 'VISp', 'AUDp']
    for reg in regions_compare:
        match = pooled_df[pooled_df['acronym'].str.contains(reg) == True]
        # all_df = match.iloc[np.where((match['names'].str.contains('2/3') == True) | (match['names'].str.contains('4') == True))[0]]

        all_df = match.iloc[np.where((match['names'].str.contains('4') == True))[0]]
        
        
        sum_df = all_df.groupby(['dataset']).sum()
    
        sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
        sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
        sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
    
        tstat, p = stats.ttest_ind(motor_df['density_W'], sum_df['density_W'], equal_var=True, alternative='two-sided')
        print(p)
        print('motor:' + str(np.mean(motor_df['density_W'])) + ' ' + reg + str(np.mean(sum_df['density_W'])))
        print(stats.sem(sum_df['density_W']))   
        
      

    ### Separate comparison L6-ORB, L6-MOs, L6-SSp vs. L6-RSP
    match = pooled_df[pooled_df['acronym'].str.contains('RSP') == True]
    RSP_df = match[match['names'].str.contains('6') == True]
  
    
    RSP_sum = RSP_df.groupby(['dataset']).sum()
    RSP_sum['density_L'] = RSP_sum['num_OLs_L']/RSP_sum['atlas_vol_L']
    RSP_sum['density_R'] = RSP_sum['num_OLs_R']/RSP_sum['atlas_vol_R']
    RSP_sum['density_W'] = RSP_sum['num_OLs_W']/RSP_sum['atlas_vol_W']
    print(stats.sem(RSP_sum['density_W'])) 
    
    match = pooled_df[pooled_df['acronym'].str.contains('ORB') == True]
    all_df = match[match['names'].str.contains('6') == True]
    
    sum_df = all_df.groupby(['dataset']).sum()
    sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
    sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
    sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
    

    tstat, p = stats.ttest_ind(RSP_sum['density_W'], sum_df['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('RSP:' + str(np.mean(RSP_sum['density_W'])) + ' ORB ' + str(np.mean(sum_df['density_W'])))
    print(stats.sem(sum_df['density_W']))   
    
      
    match = pooled_df[pooled_df['acronym'].str.contains('SSp') == True]
    all_df = match[match['names'].str.contains('6') == True]
    
    sum_df = all_df.groupby(['dataset']).sum()
    sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
    sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
    sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
    

    tstat, p = stats.ttest_ind(RSP_sum['density_W'], sum_df['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('RSP:' + str(np.mean(RSP_sum['density_W'])) + ' SSp ' + str(np.mean(sum_df['density_W'])))
    print(stats.sem(sum_df['density_W']))   
    
    
    match = pooled_df[pooled_df['acronym'].str.contains('MOs') == True]
    all_df = match[match['names'].str.contains('6') == True]
    
    sum_df = all_df.groupby(['dataset']).sum()
    sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
    sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
    sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
    

    tstat, p = stats.ttest_ind(RSP_sum['density_W'], sum_df['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('RSP:' + str(np.mean(RSP_sum['density_W'])) + ' MOs ' + str(np.mean(sum_df['density_W'])))
    print(stats.sem(sum_df['density_W']))   
    


    
    
    
    
    
    
    
    
    
    #%%% PLOT global regions of interest
    

        
        
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUD', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VIS',                                             # occipital lobe
                     # 'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg', 'RHP', 'ENT',   ### Hippocampus

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global_LR(pooled_df, names_to_plot, [palette[0], palette[3]], sav_fold, exp_name + '_GLOBAL_COMPARISON_low',
                   ylim=15000, figsize=(4.4, 3))

    names_to_plot = [
                    'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg', 'RHP', 'ENT',   ### Hippocampus
                     'MOB', 'PIR', 'COA', 'PAA', 'NLOT', 'TR',  ### Olfactory areas, Pririform-Amygdalar, Cortical-amygdalar, Postpiriform Transition area (TR), Nuclus of olfactory Tract (NLOT)
                     
                     'PAL', 'PALv', 'PALm', 'PALc', 'GPe', 'GPi',   ### Pallidum
                     'STR', 'CP', 'ACB', 'LSX', 'sAMY',   ### ACB --- nucleus acumbens, Striatum-like amygdalar nuclei (SAMY)
                     'TH','DORpm', 'DORsm', 'VAL', 'VM', 'MG', 'LGd',
                     'HY',    # Hypothalamus, contains ME (Medial eminence)
                     # 'MBsen', 'MBmot', 'MBsta',   ### sensory IC/SC, motor, behavioral
                     'SCs', 'SCm',  'IC', 'SNr', 'VTA', 'PAG',    ###'SNc', SC superior colliculus motor (m) or sensory (s) or compact (c), PAG periaqueductal gray
                     
                     
                     # 'P',   # Pons and Medulla --- currently skipped
                     'CB', 'VERM', 'HEM', 'CBN',     # Cebreellum, CBN - cerebellar nuclei
                      # 'cc', 'fxs', 'arb'   #'fiber tracts'  arb == arbor vitae in cerebellar related fiber tracts (cbf)
                     ]
    plot_global_LR(pooled_df, names_to_plot, [palette[0], palette[3]], sav_fold, exp_name + '_GLOBAL_COMPARISON_high',
                   ylim=60000, figsize=(8, 4))



    ### SEPARATE MAP COMPARING ALL "nuclei"?
    names_to_plot = [

                     ]    
    
    
    
    
    

#%% LARGE OL pooling:
if exp_name == 'LARGE' or plot_all:
    exp_name = 'LARGE'
    ### Exclude M229 --- mostly just for large OL counting since stitching was only translational
    
    ### Also exclude M223 --- very high counts, likely due to high expansion
    
    ### also weird stuff with M271 and Otx6??? --- need to retrain and re-run
    large_df = df_concat[~df_concat['dataset'].isin(['M223', 'M271', 'Otx6', 'M281', 'M260'])]
    
    rik_df_only = df_concat[df_concat['exp'].isin(['LncOL1', 'LncOL1_bloody'])]
    large_df_with_96 = large_df[large_df['exp'].isin(['P60', 'LncOL1'])]
    large_df = large_df[large_df['exp'].isin(['P60', 'P120', 'P240', 'P620', 'P800', 'P60_NEW', 'P240_NEW'])]

    

    # exp_name = 'Cuprizone'
    LARGE_cup_df = df_concat[df_concat['exp'].isin(['P60', 'Cuprizone', 'Recovery', 'Cuprizone_NEW', 'Recovery_NEW'])]
    
    ### Declare plotting variables
    palette = sns.color_palette("Set2")
    

    
    
    
    
    
    
    
    
    #%%% PLOT global regions of interest

    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', #'AUDd', 'AUDpo', 'AUDv', 
                     'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', #'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(large_df, names_to_plot, palette,  sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX',
                   dname='density_LARGE_W_CLEAN', dropna=True,
                   ylim=300, figsize=(7, 2.6))
    
    
    names_to_plot = [
                     'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg',   ### Hippocampus
                      'ENT', 'ENTl', 'ENTm', #'ENTmv',    ### Entorhinal areas divided into layer 1 - 6!!!
                     ]

    plot_global(large_df, names_to_plot, palette, sav_fold,  exp_name + '_GLOBAL_COMPARISON_HIPPO',
                   dname='density_LARGE_W_CLEAN', dropna=True,
                   ylim=300, figsize=(4, 3))
    
    

    
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', #'AUDd', 'AUDpo', 'AUDv', 
                     'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', #'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(LARGE_cup_df, names_to_plot, palette,  sav_fold, exp_name + '_CUPRIZONE_GLOBAL_COMPARISON_CORTEX',
                   dname='density_LARGE_W_CLEAN', dropna=True,
                   ylim=1000, figsize=(7, 3))
    
    
    names_to_plot = [
                     'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg',   ### Hippocampus
                      'ENT', 'ENTl', 'ENTm', #'ENTmv',    ### Entorhinal areas divided into layer 1 - 6!!!
                     ]

    plot_global(LARGE_cup_df, names_to_plot, palette, sav_fold,  exp_name + '_CUPRIZONE_GLOBAL_COMPARISON_HIPPO',
                   dname='density_LARGE_W_CLEAN', dropna=True,
                   ylim=1500, figsize=(4, 3))
    
    

    
    
    
    
    #%%% Plot by layer
    P60_only_LARGE = df_concat[df_concat['exp'].isin(['P60'])]
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
        
    plot_by_layer(P60_only_LARGE, [], names_to_plot, sav_fold, exp_name, plot_name='P60_LARGE', figsize=(4.8,4),
                  dname='density_LARGE_W_CLEAN', ylim=[0, 400])
    
    P60_only_LARGE = df_concat[df_concat['exp'].isin(['P240'])]
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
        
    plot_by_layer(P60_only_LARGE, [], names_to_plot, sav_fold, exp_name, plot_name='P240_LARGE', figsize=(4.8,4),
                  dname='density_LARGE_W_CLEAN', ylim=[0, 200])
    
    
    cup_only_LARGE = df_concat[df_concat['exp'].isin(['Cuprizone'])]
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
        
    plot_by_layer(cup_only_LARGE, [], names_to_plot, sav_fold, exp_name, plot_name='CUPRIZONE_LARGE', figsize=(4.2,3.2),
                  dname='density_LARGE_W_CLEAN', ylim=[0, 50])

    rec_only_LARGE = df_concat[df_concat['exp'].isin(['Recovery'])]
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
        
    plot_by_layer(rec_only_LARGE, [], names_to_plot, sav_fold, exp_name, plot_name='RECOVERY_LARGE', figsize=(4.2,3.2),
                  dname='density_LARGE_W_CLEAN', ylim=[0, 2000])



    
    
    #%%% Get rank order of areas with highest density of large OLs in old  brains!
    
    df_means = large_df.groupby(large_df.index).mean(numeric_only=True)
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
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, large_df, reg_name='Isocortex', dname='density_LARGE_W_CLEAN', 
                                                    lvl_low=5, lvl_high=9, to_remove=to_remove)
    
    plot_rank_order(plot_vals, exp_name='P800', dname='density_LARGE_W_CLEAN', 
                    figsize=(3.5,4))    
    
    plot_rank_order(plot_vals, exp_name='P620', dname='density_LARGE_W_CLEAN', 
                    figsize=(3.5,4))    
    
    plot_rank_order(plot_vals, exp_name='P60', dname='density_LARGE_W_CLEAN', 
                    figsize=(3.5,4))    
    
        
    
    
    

    #%%% Try plotting and normalizing to LncOL1
 
    
    # set density_W to be density_LARGE for comparisons!
    # large_df_with_96.loc[large_df_with_96['exp'] == 'LncOL1', 'density_LARGE_W_CLEAN'] = large_df_with_96[large_df_with_96['exp'].isin(['LncOL1'])]['density_W']
    # df_mean_rik = large_df_with_96.groupby(large_df_with_96.index).mean(numeric_only=True)
    # df_mean_rik['acronym'] = keys_df['acronym']
    # df_mean_rik['dataset'] = keys_df['dataset']
    # df_mean_rik['names'] = keys_df['names']
    # df_mean_rik['children'] = keys_df['children']  
    
    
    # plot_vals = plot_stripes_from_df(df_mean_rik, large_df_with_96, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Isocortex', dname='density_W',
    #             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=(6,8), lvl_low=5, lvl_high=9, leg_loc='lower right',
    #             x_lab='Density of LncOL1+ cells (cells/mm\u00b3)', y_lab='', xlim=[0, 2000], name='medium_level') 
    
    
    
    
    ### Plot regression
    
    to_remove = 'VISpl'
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, large_df_with_96, reg_name='Isocortex', dname='density_LARGE_W_CLEAN', to_remove=to_remove, lvl_low=5, lvl_high=9)
    
    all_avg = []
    for i_n, reg_name in enumerate(names_to_plot):
        match = plot_vals[plot_vals['acronym'].str.fullmatch(reg_name) == True]
    
        # baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
        # match['fold_change'] = match['density_W']/baseline
        
        ### Also relate beginning density to fold change?
        avg_stats = match.groupby(['exp', 'acronym']).mean(numeric_only=True).reset_index()
        # copy the density of LncOL1 cells over so can plot altogether later!
        avg_stats.loc[np.where(avg_stats['exp'] == 'P60')[0], 'density_W'] = avg_stats.loc[np.where(avg_stats['exp'] == 'LncOL1')[0], 'density_W'].values[0]
         
        avg_stats = avg_stats.loc[np.where(avg_stats['exp'] == 'P60')[0]]
        
        all_avg.append(avg_stats)
    
    
    all_avg = pd.concat(all_avg)
        
    
    
        
    plt.figure(figsize=fig_long)
    fig = sns.regplot(data=all_avg, x='density_W', 
                                    y='density_LARGE_W_CLEAN', color='gray')

    r,p = stats.pearsonr(all_avg.dropna()['density_W'], all_avg.dropna()['density_LARGE_W_CLEAN'])
    print(r)
    
    
    # vals = stats.linregress(all_avg.dropna()['density_W'], all_avg.dropna()['density_LARGE_W'])  ### THIS CAN ALSO GIVE YOU SLOPE
    
    
    all_texts = []
    for line in range(0,all_avg.shape[0]):
         all_texts.append(plt.text(all_avg['density_W'].iloc[line]+0.01, all_avg['density_LARGE_W_CLEAN'].iloc[line], 
         all_avg['acronym'].iloc[line],
         ha='center', va='center',
         size=10, color='black', weight='normal'))
    #plt.text(all_texts)
    
    ax = plt.gca()
    ax.legend().set_visible(False)
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #plt.legend(loc = 'upper left', frameon=False)
    # plt.ylim([0, 30])
    plt.xlabel('Density of LncOL1+ cells (cells/mm\u00b3)', fontsize=fontsize)
    plt.ylabel('Density of large OLs (cells/mm\u00b3)', fontsize=fontsize)
    plt.tight_layout()
    
    adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.savefig(sav_fold + exp_name + '_density_P60_vs_LncOL1.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_density_P60_vs_LncOL1.svg', format='svg', dpi=300)
    
    
    

    
    
    
    
    #%%% LncOL1 only
    df_mean_rik = rik_df_only.groupby(rik_df_only.index).mean(numeric_only=True)
    df_mean_rik['acronym'] = keys_df['acronym']
    df_mean_rik['dataset'] = keys_df['dataset']
    df_mean_rik['names'] = keys_df['names']
    df_mean_rik['children'] = keys_df['children']   
    
    
    plot_vals = plot_stripes_from_df(df_mean_rik, rik_df_only, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Isocortex', dname='density_W',
                sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=9, leg_loc='lower right',
                x_lab='Density of LncOL1+ cells (cells/mm\u00b3)', y_lab='', xlim=[0, 1000], name='medium_level') 
    
    plt.savefig(sav_fold + exp_name + '_density_LncOL1_only.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_density_LncOL1_only.svg', format='svg', dpi=300)
    
        



#%% AGING pooling

if exp_name == 'Aging' or plot_all:
    
    exp_name = 'Aging'
    
    ### Exclude M229 --- mostly just for large OL counting since stitching was only translational
    ### Included for now... just for fun
    # aging_df = df_concat[~df_concat['dataset'].isin(['M229'])]
    aging_df = df_concat
    
    aging_df = aging_df[aging_df['exp'].isin(['P60', 'P120', 'P240', 'P360', 'P620', 'P100', 'P800', 'P60_NEW', 'P60_NEW_NEW', 'P60_NEW_ORIG',
                                              'P120_NEW', 'P120_NEW_NEW', 'P240_NEW', 'P240_NEW_NEW','P620_NEW', 'P620_NEW_NEW', 'P800_NEW', 'P800_NEW_NEW'])]
    
    df_means = aging_df.groupby(aging_df.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
   
    
 
    #%%% Compare GLOBAL regions
     
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
    palette = sns.color_palette("husl", len(names_to_plot))
    
    palette[-1] =sns.color_palette("Set2", len(names_to_plot))[5]
    
    compare_GLOBAL_regions(aging_df, names_to_plot, palette, xlim=[0, 850], ylim=60000, ylim_norm=[1,2.2], sav_name='AGING', sav_fold=sav_fold, exp_name=exp_name, 
                           fontsize=14, add_zero=True, figsize=(4,3))
    
    
    # Compare fiber tracts P60 vs. P640
    for name in names_to_plot:
        match = aging_df[aging_df['acronym'].str.fullmatch(name) == True]
        P60 = match[match['age'] == 60]
        P620 = match[match['age'] == 620]    
        print('Change in density: ' + str(np.mean(P620['density_W']) - np.mean(P60['density_W'])) + name)
    
    # also compare fold change
    for name in names_to_plot:
        match = aging_df[aging_df['acronym'].str.fullmatch(name) == True]
        P60 = match[match['age'] == 60]
        P620 = match[match['age'] == 620]    
        print('Change in density: ' + str(np.mean(P620['density_W'])/np.mean(P60['density_W'])) + name)
    
                
    
    
    
    # motor_df = match[match['names'].str.contains('2/3') == True]
    # print(stats.sem(P60['density_W']))   
    # print(stats.sem(P620['density_W']))   
    

    # tstat, p = stats.ttest_ind(P60['density_W'], P620['density_W'], equal_var=True, alternative='two-sided')
    # print(p)
    # print('P60:' + str(np.mean(P60['density_W'])) + ' P620: ' + str(np.mean(P620['density_W'])))


    ### Also compare ratio of primary vs. secondary area
    
    match = aging_df[aging_df['acronym'].str.fullmatch('SSp') == True]
    P60_SSp = match[match['age'] == 60]
    P620_SSp = match[match['age'] == 620]    
    
    match = aging_df[aging_df['acronym'].str.fullmatch('SSs') == True]
    P60_SSs = match[match['age'] == 60]
    P620_SSs = match[match['age'] == 620]    
        
    print('Primary vs. Secondary at P60: ' + str(np.mean(P60_SSp['density_W'])/np.mean(P60_SSs['density_W'])) + 'SSp P60')
    print('Primary vs. Secondary at P620: ' + str(np.mean(P620_SSp['density_W'])/np.mean(P620_SSs['density_W'])) + 'SSp P620')

    match = aging_df[aging_df['acronym'].str.fullmatch('MOp') == True]
    P60_MOp = match[match['age'] == 60]
    P620_MOp = match[match['age'] == 620]    
    
    match = aging_df[aging_df['acronym'].str.fullmatch('MOs') == True]
    P60_MOs = match[match['age'] == 60]
    P620_MOs = match[match['age'] == 620]    
        
    print('Primary vs. Secondary at P60: ' + str(np.mean(P60_MOp['density_W'])/np.mean(P60_MOs['density_W'])) + 'MOp P60')
    print('Primary vs. Secondary at P620: ' + str(np.mean(P620_MOp['density_W'])/np.mean(P620_MOs['density_W'])) + 'MOs P620')


    
    ### Compare Hippocampal Dentate Gryus across timepoints
    DG_po_df = aging_df[aging_df['acronym'].str.fullmatch('DG-po') == True]
    DG_mo_df = aging_df[aging_df['acronym'].str.fullmatch('DG-mo') == True]
    DG_sg_df = aging_df[aging_df['acronym'].str.fullmatch('DG-sg') == True]
    
    ages = [60, 240, 620, 850]
    for age in ages:
        DG_po = DG_po_df[DG_po_df['age'] == age]
        DG_mo = DG_mo_df[DG_mo_df['age'] == age]    
        DG_sg = DG_sg_df[DG_sg_df['age'] == age]
        
        # print(stats.sem(DG_po['density_W']))   
        # print(stats.sem(DG_mo['density_W']))  
        # print(stats.sem(DG_sg['density_W']))  
    
        tstat, p = stats.ttest_ind(DG_po['density_W'], DG_mo['density_W'], equal_var=True, alternative='two-sided')
        print(p)
        print(str(age) + '  ' + str(np.mean(DG_po['density_W'])) + '   ' + str(np.mean(DG_mo['density_W'])))
        
        tstat, p = stats.ttest_ind(DG_po['density_W'], DG_sg['density_W'], equal_var=True, alternative='two-sided')
        print(p)
        print(str(age) + '  ' + str(np.mean(DG_po['density_W'])) + '   ' + str(np.mean(DG_sg['density_W'])))


    
    



    #%% Get df Layers
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    
    # names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    
    layers = ['1', '2/3', '4', '5', '6']
    
    fontsize = 14
    
    #%%% Compare layers for each cortical region
    all_layers = []
    for layer in layers:
        plt.figure(figsize=fig_long)
        for i_n, name in enumerate(names_to_plot):
            # if name matches the first half at minimum, then go and plot
            match = aging_df[aging_df['acronym'].str.contains(name) == True]
            if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
                match = match[match['acronym'].str.contains('RSPd4') == False]   
            match=match.reset_index()
            
    
            ### DROP layer 1 from M127 and M126 for the moment due to delipidation artifiact
            if layer == '1':
                match = match.drop(index=np.where(match['dataset'] == 'M127')[0]).reset_index()        
                match = match.drop(index=np.where(match['dataset'] == 'M126')[0])
    
            lay_df = match[match['names'].str.contains(layer) == True]
 
    #%%% Compare GLOBAL regions
     
            
            sum_df = lay_df.groupby(['dataset', 'exp', 'age']).sum(numeric_only=True)
            sum_df = sum_df.reset_index() ### moves all of the indices into columns!!!
    
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
    
            sns.lineplot(x=sum_df['age'], y=sum_df['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)]   )
    
            plt.title('Layer ' + layer)
    
            # give name of layer
            sum_df['layer'] = layer
        
            all_layers.append(sum_df)
            
    df_layers = pd.concat(all_layers)
    
    df_layers = df_layers.groupby(['dataset', 'layer', 'age']).sum(numeric_only=True)     
    df_layers['density_W'] = df_layers['density_W']/df_layers['atlas_vol_W']   ### scale it down since it's sum above   
    
    df_layers = df_layers.reset_index()
    
    #%%% Plot layers over time
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    # names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    fontsize = 14
    
    layer_colors = ['deepskyblue',
                    'blueviolet',
                    'forestgreen',
                    'goldenrod',
                    'grey'
                    ]
    layers = ['1', '2/3', '4', '5', '6']
    plt.figure(figsize=(3.5, 3))
    for i_n, layer in enumerate(layers):
        match = df_layers[df_layers['layer'].str.contains(layer) == True]
        baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
        
        match['fold_change'] = match['density_W']/baseline
        
        sns.lineplot(x=match['age'], y=match['fold_change'], label=layer, color=layer_colors[i_n % len(layer_colors)], linestyle=styles[i_n % len(styles)],
                      alpha=0.8,
                      errorbar='se')
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    plt.legend(loc = 'upper left', frameon=False)
    plt.ylim([1, 3])
    plt.xlabel('Age (days)', fontsize=fontsize)
    plt.ylabel('Density fold change', fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(sav_fold + exp_name + '_LAYERS_over_time.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_LAYERS_over_time.svg', format='svg', dpi=300)
    
    ### Plt RAW DENSITY of OLs over time
    layer_colors = ['deepskyblue',
                    'blueviolet',
                    'forestgreen',
                    'goldenrod',
                    'grey'
                    ]
    plt.figure(figsize=(3.5, 3))
    for i_n, layer in enumerate(layers):
        match = df_layers[df_layers['layer'].str.contains(layer) == True]
        baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
        
        match['raw_change'] = match['density_W'] - baseline
        
        sns.lineplot(x=match['age'], y=match['raw_change'], label=layer, color=layer_colors[i_n % len(layer_colors)], linestyle=styles[i_n % len(styles)],
                      alpha=0.8,
                      errorbar='se')
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    #ax.spines['left'].set_visible(False)
    plt.legend(loc = 'upper left', frameon=False)
    plt.ylim([0, 6000])
    plt.xlabel('Age (days)', fontsize=fontsize)
    plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(sav_fold + exp_name + '_LAYERS_DENSITY_CHANGE_over_time.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_LAYERS_DENSITY_CHANGE_over_time.svg', format='svg', dpi=300)
    
       
    
    #%%% Plot by time for cortical regions
    to_remove = 'MO|SS|AUDp|AUDd|VISpl|VISrl|AUDv|VISal|VISp|VISl|VISa|AUDpo|VISpor|VISam|VISli|VISpm'
    
    to_remove = to_remove + '|ECT|PERI|PL'  ### still some registration issues at 4mos
    
    to_remove = to_remove + '|ORB|GU|ACA|PTLp|Al|FRP|ILA'
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, aging_df, reg_name='Isocortex', dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=9)
    
    all_avg = []
    plt.figure(figsize=(3.5, 3))
    for i_n, reg_name in enumerate(names_to_plot):
        match = plot_vals[plot_vals['acronym'].str.fullmatch(reg_name) == True]
        baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
        
        match['fold_change'] = match['density_W']/baseline
        
        sns.lineplot(x=match['age'], y=match['fold_change'], label=reg_name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                     errorbar='se')
        
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    plt.legend(loc = 'upper left', frameon=False)
    plt.ylim([1, 3])
    plt.xlabel('Age (days postnatal)', fontsize=fontsize)
    plt.ylabel('Density fold change', fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(sav_fold + exp_name + '_CORTICAL_REGIONS_over_time.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_CORTICAL_REGIONS_over_time.svg', format='svg', dpi=300)
    
    
    
    
    #%%% plot scatterplot comparing starting density and final fold change
    #to_remove = 'MO|SS|AUDp|AUDd|VISpl|VISrl|AUDv|VISal|VISp|VISl|VISa|AUDpo|VISpor|VISam|VISli|VISpm'
    #to_remove = to_remove + '|ECT|PERI|PL'  ### still some registration issues at 4mos
    to_remove = 'VISpl'
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, aging_df, reg_name='Isocortex', dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=9)
    
    all_avg = []
    for i_n, reg_name in enumerate(names_to_plot):
        match = plot_vals[plot_vals['acronym'].str.fullmatch(reg_name) == True]
        baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
        match['fold_change'] = match['density_W']/baseline
        
        ### Also relate beginning density to fold change?
        avg_stats = match.groupby(['exp', 'acronym']).mean(numeric_only=True).reset_index()
        # copy the fold change over from 23mos to P60 so can plot later
        avg_stats.loc[np.where(avg_stats['exp'] == 'P60')[0], 'fold_change'] = avg_stats.loc[np.where(avg_stats['exp'] == 'P620')[0], 'fold_change'].values[0]
         
        avg_stats = avg_stats.loc[np.where(avg_stats['exp'] == 'P60')[0]]
        
        all_avg.append(avg_stats)
    
    
    all_avg = pd.concat(all_avg)
    
    
    plt.figure(figsize=(3.5,3))
    fig = sns.regplot(data=all_avg, x='density_W', y='fold_change', color='gray')
    
    #calculate slope and intercept of regression equation
    slope, intercept, r, p, sterr = stats.linregress(x=fig.get_lines()[0].get_xdata(),
                                                           y=fig.get_lines()[0].get_ydata())
    
    #display slope and intercept of regression equation
    print(slope)

    r,p = stats.pearsonr(all_avg.dropna()['density_W'], all_avg.dropna()['fold_change'])
    print(r)
    print(p)
    
    
    all_texts = []
    for line in range(0,all_avg.shape[0]):
         all_texts.append(plt.text(all_avg['density_W'].iloc[line]+0.01, all_avg['fold_change'].iloc[line], 
         all_avg['acronym'].iloc[line],
         ha='center', va='center',
         size=10, color='black', weight='normal'))
    #plt.text(all_texts)
    
    ax = plt.gca()
    ax.legend().set_visible(False)
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #plt.legend(loc = 'upper left', frameon=False)
    plt.ylim([1, 3])
    plt.xlabel('Density at P60 (cells/mm\u00b3)', fontsize=fontsize)
    plt.ylabel('Fold change at P650', fontsize=fontsize)
    
    ### set scientific notation
    ax.ticklabel_format(axis='x', scilimits=(-4, 4))
    ax.xaxis.get_offset_text().set_fontsize(fontsize-2)
    plt.xticks(np.arange(0, 15000, step=5000))
    
    
    
    plt.tight_layout()
    
    adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.savefig(sav_fold + exp_name + '_density_P60_vs_fold_change.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_density_P60_vs_fold_change.svg', format='svg', dpi=300)
    
    
    ### Declare plotting variables
    palette = sns.color_palette("Set2")
    ### CAREFUL ---> df_means right now is pooled from ALL groups... for sorting...
    
    
 
    #%%% Compare layers for each cortical region
    # names_to_plot = ['SSp', 'MOp','RSP', 'SSs', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'TEa']
    # names_to_plot = ['ACA','FRP', 'GU', 'ILA', 'PL', 'AI',  'VISC']  # 'PTLp' doesn't exist?
    names_to_plot = ['SSp', 'MOp','RSP', 'SSs', 'MOs', 'VIS', 'AUD', 'PERI', 'TEa', 'PL', 'ACA', 'AI']

    palette = sns.color_palette("Set2", len(names_to_plot))
    #styles = ['-', '--', '-.', ':']
    styles=['-']
    
    layers = ['1', '2/3', '4', '5', '6']
    
    fontsize = 14
    
    # all_layers = []

    fig1, ax1 = plt.subplots(4, 3, figsize=(8, 8), sharex=True, sharey=True)
    fig2, ax2 = plt.subplots(4, 3, figsize=(8, 8), sharex=True, sharey=True)
    for i_x, name in enumerate(names_to_plot):

        for i_n, layer in enumerate(layers):
            # if name matches the first half at minimum, then go and plot
            match = aging_df[aging_df['acronym'].str.contains(name) == True]
            if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
                match = match[match['acronym'].str.contains('RSPd4') == False]   
            match=match.reset_index()
            

            ### DROP layer 1 from M127 and M126 for the moment due to delipidation artifiact
            # if layer == '1':
                #match = match.drop(index=np.where(match['dataset'] == 'M267')[0]).reset_index()
            #     #match = match.drop(columns=['level_0'])
            #     match = match.drop(index=np.where(match['dataset'] == 'M127')[0]).reset_index()
            #     match = match.drop(columns=['level_0'])
            #     match = match.drop(index=np.where(match['dataset'] == 'M126')[0]).reset_index()
            #     #match = match.drop(index=np.where(match['dataset'] == 'M310')[0])     

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                    
            lay_df = match[match['names'].str.contains(layer) == True]
            
            if len(lay_df) == 0:
                continue

            zero_df = {'exp':'P0', 'age':0, 'density_W':0, 'density_NORM':0, 'num_OLs_W':0, 'atlas_vol_W':1}
            zero_df = pd.DataFrame(zero_df, index=[5])
            
            lay_df = pd.concat([lay_df, zero_df]).fillna(0)
            
            
    
            cur_ax = sns.lineplot(ax=ax1.flatten()[i_x], x=lay_df['age'], y=lay_df['density_W'], label=layer, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                         errorbar='se',
                         marker='o',
                         markersize=8, linewidth=2).set(title=name) 
            
            lay_df['density_NORM'] = lay_df['density_W']/np.nanmean(lay_df.iloc[np.where(lay_df['exp'] == 'P60')[0]]['density_W'])


            sns.lineplot(ax=ax2.flatten()[i_x], x=lay_df['age'], y=lay_df['density_NORM'], label=layer, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                         errorbar='se',
                         marker='o',
                         markersize=8, linewidth=2).set(title=name) 
            ax2.flatten()[i_x].set_ylim(0, 6)
            
            
            
    ### Remove legends
    for ax in ax1.flatten():
        ax.legend([],[], frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        ax.set_ylim([0, 60000])
        ax.ticklabel_format(axis='y', scilimits=(-4, 4))
        ax.xaxis.get_offset_text().set_fontsize(fontsize-2)
        
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        
    for ax in ax2.flatten():
        ax.legend([],[], frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('')
        ax.set_ylabel('')

        
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        
        
    handles, labels = ax1[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right')
    fig2.legend(handles, labels, loc='upper right')


    
    fig1.text(0.04, 0.5, 'Density (cells/mm\u00b3)', va='center', rotation='vertical', fontsize=fontsize)
    fig2.text(0.04, 0.5, 'Normalized cell density', va='center', rotation='vertical', fontsize=fontsize)

    plt.figure(fig1)
    plt.savefig(sav_fold + exp_name +'_by_layer_and_region_AGING.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_by_layer_and_region_AGING.svg', format='svg', dpi=300)
    

    plt.figure(fig2)
    plt.savefig(sav_fold + exp_name +'_by_layer_and_region_AGING_NORM.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_by_layer_and_region_AGING_NORM.svg', format='svg', dpi=300)
    
    
    
    
    #%%% PLOT global regions of interest

    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(aging_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX',
                   ylim=25000, figsize=(7, 3))
    
    
    # plot raw numbers
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(aging_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX_RAW_COUNT', dname='num_OLs_W',
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
                       # 'SUB', #'SUBv', 'SUBd'   ### subiculum

                     ]

    plot_global(aging_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_HIPPO',
                   ylim=25000, figsize=(4.2, 2.5))
    
    
    ### stats for comparison
    # Compare Entorhinal areas
    match = aging_df[aging_df['acronym'].str.fullmatch('ENTl') == True]
    ENTL60 = match[match['age'] == 60]
    ENTL620 = match[match['age'] == 620]
        
    match = aging_df[aging_df['acronym'].str.fullmatch('ENTm') == True]
    ENTm60 = match[match['age'] == 60]    
    ENTm620 = match[match['age'] == 620]    
    
    
    print(stats.sem(ENTL60['density_W']))   
    print(stats.sem(ENTL620['density_W']))   
    print(stats.sem(ENTm60['density_W']))   
    print(stats.sem(ENTm620['density_W']))   
    

    tstat, p = stats.ttest_ind(ENTL60['density_W'], ENTm60['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('ENTL60:' + str(np.mean(ENTL60['density_W'])) + ' ENTm60: ' + str(np.mean(ENTm60['density_W'])))

    tstat, p = stats.ttest_ind(ENTL620['density_W'], ENTm620['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('ENTL620:' + str(np.mean(ENTL620['density_W'])) + ' ENTm620: ' + str(np.mean(ENTm620['density_W'])))
  
    
    
        
        
    
        
    names_to_plot = [
                     'CBX', 'VERM', #vermal regions ### Cerebellum

                        ### Hemispheric regions, 
                      'HEM', 'SIM', 'AN', 'PRM', 'COPY', 'PFL', 'FL',    
                      
                      # 'FN', 'IP', 'DN',   ### Cerebellar nuclei
                      # 'arb',   ### arbor vitae

                     ]

    plot_global(aging_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CEREBELLUM',
                   ylim=25000, figsize=(4.2, 2.5))
    
    
    


        
    