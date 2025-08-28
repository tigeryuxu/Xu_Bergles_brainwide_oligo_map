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




#%% Also load in control brain --- OPTIONAL
# list_brains = get_metadata(mouse_num = ['M296'])
# input_path = list_brains[0]['path']
# name_str = list_brains[0]['name']
# n5_file = input_path + name_str + '.n5'
# downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
# atlas_dir_c1 = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
# f = z5py.File(n5_file, "r")
# dset_c1 = f['setup0/timepoint0/s0']  




#%% 5xFAD

list_brains = get_metadata(mouse_num = [
                                        'M243',
                                        'M234', 
                                        'M244', 
                                        'M235', 
                                        # 'M217'
                                        ]) ### 5xFAD
                                     #   'M304', 'M242', 'M297', 'M296'])        ### control



sav_fold = '/media/user/4TB_SSD/Plot_outputs_HALO/'

cloudreg = 0

ANTS = 1

#%% Parse the json file so we can choose what we want to extract or mask out
reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'


ref_atlas = tiff.imread(reference_atlas)
ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)


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

all_dfs = []
for exp in list_brains:


    input_path = exp['path']
    name_str = exp['name']
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
    
    
    f = z5py.File(n5_file, "r")
    dset = f['setup0/timepoint0/s0']
    
    dset_halo = f['setup1/timepoint0/s0']

    dset_congo = f['setup2/timepoint0/s0']    

    scaled = 1

    allen_dir = downsampled_dir + name_str + '_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
    analysis_dir = input_path + name_str + '_postprocess_CONGO/'
    

    myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_n4_down1_resolution_20_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
    auto_path = glob.glob(os.path.join(downsampled_dir, '*_ch1_n4_down1_resolution_20_PAD.tif'))[0]
    
 
    pad = True
    
    XY_res = 1.152035240378141
    Z_res = 5
    
    
    res_diff = XY_res/Z_res
    
    ### Initiate poolThread
    #poolThread_load_data = ThreadPool(processes=1)
    
    """ Loop through all the folders and do the analysis!!!"""
    #filename = n5_file.split('/')[-2]
    filename = n5_file.split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename) 
    sav_dir = input_path + '/' + filename + '_postprocess_CONGO'
    # sav_dir = input_path + '/' + filename + '_postprocess_60' #updated the folder name to include the new threshold
    
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
    

    
    
    #%% SAVE coords_df
    
    import pickle as pkl


    #%% KEEP LOADING UNTIL HAVE ALL OF THEM

    # # Load pickle
    with open(sav_dir + 'coords_df_HALO_data', 'rb') as file:
        coords_df = pkl.load(file)
        
        
    coords_df['exp'] = exp['num']

    all_dfs.append(coords_df)



coords_df = pd.concat(all_dfs)




#%% Make a mean column from across all control brains
sixmos_cols = [col for col in coords_df.columns if '6mos' in col]

# Replace 0 and -1 with NaN, then take row-wise mean
masked = coords_df[sixmos_cols].replace({0: np.nan, -1: np.nan})
coords_df['mean_6mos_myelin'] = masked.mean(axis=1)


control_col = 'mean_6mos_myelin' ### if want to use pooled
# control_col = 'M296_MoE_6mos_fused_myelin'

# control_col = 'M297_MoE_6mos_fused_myelin'
# control_col = 'M304_MoE_6mos_fused_myelin'

# control_col = 'M242_MoE_control_6mos_fused_dataset_myelin'

zzz

#%% Filter to find which are in fimbria vs. cortex

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu

# For re-running code, make sure to drop region group
if 'region_group' in coords_df.columns:
    coords_df = coords_df.drop(columns='region_group')


palette = sns.color_palette("Set2")[3:5]
mpl.rcParams['svg.fonttype'] = 'none'

### Get fimbria
wm_name = 'fimbria'
white_matter_ids = get_sub_regions_atlas(main_keys, child_id=[], sub_keys=[], reg_name=wm_name)
white_matter_ids = main_keys.iloc[white_matter_ids]['ids'].tolist()

### Get cortex
cortex_name = 'Isocortex'
cortical_ids = get_sub_regions_atlas(main_keys, child_id=[], sub_keys=[], reg_name=cortex_name)
cortical_ids = main_keys.iloc[cortical_ids]['ids'].tolist()

### Label region groups
coords_df['region_group'] = coords_df['region_ids'].apply(
    lambda rid: wm_name if rid in white_matter_ids 
    else cortex_name if rid in cortical_ids 
    else None
)

# Keep only labeled rows
grouped_df = coords_df[coords_df['region_group'].notna()].copy()
grouped_df['myelin_diff'] = grouped_df['myelin_int'] - grouped_df[control_col]

# === Style Settings ===
label_fontsize = 12
tick_fontsize = label_fontsize - 2

# Plot
fig, ax = plt.subplots(figsize=(2.5, 3))
sns.boxplot(
    data=grouped_df,
    x='region_group',
    y='myelin_diff',
    palette=palette,
    showfliers=False,
    ax=ax
)

# Format axes
ax.set_xlabel('', fontsize=label_fontsize)  # Editable label
ax.set_ylabel('Intensity diff. (5xFAD - Ctrl)', fontsize=label_fontsize)
ax.tick_params(axis='y', labelsize=tick_fontsize)
ax.tick_params(axis='x', labelsize=label_fontsize)

ax.set_ylim(top=1000, bottom=-1000)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Run stats
wm_vals = grouped_df[grouped_df['region_group'] == wm_name]['myelin_diff']
ctx_vals = grouped_df[grouped_df['region_group'] == cortex_name]['myelin_diff']
stat, p = mannwhitneyu(wm_vals, ctx_vals, alternative='two-sided')

# Annotate p-value
text = f"p = {p:.3e}"
ymax = ax.get_ylim()[1]
ax.text(0.5, ymax * 0.95, text, ha='center', va='top', fontsize=label_fontsize)




plt.tight_layout()
plt.savefig(sav_fold + "myelin_diff_boxplot.svg", dpi=300, format='svg')
plt.show()

print(f"Mann–Whitney U test: U = {stat:.2f}, p = {p:.4e}")





#%% Pair-wise plot
    
# # === Parameters ===
# sample_n = 50
# rng = np.random.default_rng(42)

# # Subsample
# subsampled_df = (
#     grouped_df
#     .groupby('region_group')
#     .apply(lambda x: x.sample(n=min(sample_n, len(x)), random_state=rng))
#     .reset_index(drop=True)
# )

# # Long-format
# long_df = subsampled_df[['region_group', 'myelin_int', control_col]].copy()
# long_df['halo_id'] = subsampled_df.index
# long_df = pd.melt(
#     long_df,
#     id_vars=['region_group', 'halo_id'],
#     value_vars=['myelin_int', control_col],
#     var_name='condition',
#     value_name='intensity'
# )
# long_df['condition'] = long_df['condition'].map({control_col: 'Control', 'myelin_int': '5xFAD'})

# long_df = long_df.sort_values(by='condition', ascending=False)   ### sort so control plotted first

# # Plot
# g = sns.FacetGrid(long_df, col='region_group', sharey=True, height=3, aspect=1)

# # Gray lines for individual halos
# def plot_lines(data, **kwargs):
#     ax = plt.gca()
#     for _, halo_df in data.groupby('halo_id'):
#         ax.plot(
#             halo_df['condition'],
#             halo_df['intensity'],
#             color='gray',
#             alpha=0.2,
#             linewidth=0.8,
#             marker='o',
#             markersize=3
#         )

#     # Colored mean line
#     region = halo_df['region_group'].iloc[0].lower()
#     set2_colors = palette
#     region_colors = {'fimbria': set2_colors[0], 'isocortex': set2_colors[1]}

#     mean_vals = data.groupby('condition')['intensity'].mean()
#     ax.plot(
#         mean_vals.index,
#         mean_vals.values,
#         color=region_colors.get(region, 'black'),
#         linewidth=2.5,
#         marker='o',
#         markersize=5
#     )

# g.map_dataframe(plot_lines)

# # Style each subplot
# for ax in g.axes.flatten():
#     ax.set_xlabel('')  # No x-label for categorical axis
#     ax.set_ylabel('Myelin Intensity', fontsize=label_fontsize)
#     ax.tick_params(axis='x', labelsize=label_fontsize)
#     ax.tick_params(axis='y', labelsize=tick_fontsize)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# plt.tight_layout()
# plt.savefig(sav_fold + "myelin_pairwise_subsampled_faceted.svg", dpi=300, format='svg')
# plt.show()
    


#%% Core volume boxplot
ymax = 14

grouped_df['core_vols_log'] = np.log1p(grouped_df['core_vols_um'])  # log(1 + x) to handle 0s


# Plot
fig, ax = plt.subplots(figsize=(2.5, 3))

sns.violinplot(
    data=grouped_df,
    x='region_group',
    y='core_vols_log',
    palette=palette,
    cut=0,
    inner='quartile',
    linewidth=1,
    ax=ax
)

# Add both raw and log medians as text annotations
categories = grouped_df['region_group'].unique()
for i, group in enumerate(categories):
    # Raw and log values for the group
    raw_vals = grouped_df[grouped_df['region_group'] == group]['core_vols_um']
    log_vals = grouped_df[grouped_df['region_group'] == group]['core_vols_log']
    
    raw_median = np.median(raw_vals)
    log_median = np.median(log_vals)

    # Compose annotation text
    annotation = (
        f"Median = {raw_median:.0f} μm³\n"
        f"log(1 + x) = {log_median:.2f}"
    )

    # Place text just below the top of the axis
    ax.text(
        i, ymax * 0.97,  # Slightly down from top
        annotation,
        ha='center',
        va='top',
        fontsize=label_fontsize
    )
    
# Axes
ax.set_xlabel('')
ax.set_ylabel(r'$\log(1 + \mathrm{Volume}\ [\mu m^3])$', fontsize=label_fontsize)
ax.tick_params(axis='x', labelsize=label_fontsize)
ax.tick_params(axis='y', labelsize=tick_fontsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Manual Y limit if desired
ax.set_ylim(top=ymax, bottom=0)


# Get log-transformed values by group
wm_vals = grouped_df[grouped_df['region_group'] == wm_name]['core_vols_log']
ctx_vals = grouped_df[grouped_df['region_group'] == cortex_name]['core_vols_log']

# Run Mann–Whitney U test
stat, p = mannwhitneyu(wm_vals, ctx_vals, alternative='two-sided')

# Annotate p-value
ymax = ax.get_ylim()[1]
ax.text(0.5, ymax * 0.8, f"p = {p:.3e}", ha='center', va='top', fontsize=label_fontsize)

print(f"Mann–Whitney U test (log core_vols): U = {stat:.2f}, p = {p:.4e}")


plt.tight_layout()
plt.savefig(sav_fold + "core_volume_violin_log_median.svg", dpi=300, format='svg')
plt.show()



#%% What percentage have zero or very small cores per experiment

# === Compute percentages per experiment
thresholds = [0, 1000]
percent_data = []

print("\nPercentage of halos with core volume ≤ threshold:")
for group in grouped_df['region_group'].unique():
    # Now loop through each experiment within the group
    for exp_id in grouped_df['exp'].unique():
        # Filter for this group and experiment
        exp_data = grouped_df[(grouped_df['region_group'] == group) & (grouped_df['exp'] == exp_id)]
        core_vols = exp_data['core_vols_um']
        total = len(core_vols)
        
        if total == 0:  # Skip if no data for this combination
            continue

        print(f"\n{group.title()} - {exp_id} (n = {total}):")
        for t in thresholds:
            count = (core_vols <= t).sum()
            percent = (count / total) * 100
            percent_data.append({
                'Region': group.title(),
                'Threshold': f"≤ {t} μm³",
                'Percent': percent,
                'Experiment': exp_id
            })
            print(f"  Volume ≤ {t} μm³: {percent:.2f}% ({count} halos)")

# Create DataFrame for plotting
percent_df = pd.DataFrame(percent_data)

# === Plot
fig, ax = plt.subplots(figsize=(4, 3))

# First create the bars (but with lower alpha)
sns.barplot(
    data=percent_df,
    x='Threshold',
    y='Percent',
    hue='Region',
    palette=palette,
    alpha=0.3,  # Make bars transparent
    ax=ax,
    zorder=3
    )

# Then add individual points for each experiment
sns.stripplot(
    data=percent_df,
    x='Threshold',
    y='Percent',
    hue='Region',
    palette=palette,
    dodge=True,  # Dodge to align with bars
    size=8,
    ax=ax,
    zorder=1
)

# Styling
ax.set_xlabel('Core volume', fontsize=label_fontsize)
ax.set_ylabel('Percent of Halos', fontsize=label_fontsize)
ax.tick_params(axis='x', labelsize=label_fontsize)
ax.tick_params(axis='y', labelsize=tick_fontsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylim(top=100, bottom=0)

# Legend
ax.legend(title=None, fontsize=label_fontsize, loc='best', frameon=False)

plt.tight_layout()
plt.savefig(sav_fold + "core_volume_threshold_barplot_by_exp.svg", dpi=300, format='svg')
plt.show()



#%% Pie charts
# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Professional color scheme with two magentas and a light green
colors = ['#E754A6',  # Bright magenta
          '#9D3C7D',  # Darker magenta
          '#7AB98E']  # Light green

# Process data for each region
for ax, region in zip([ax1, ax2], ['Isocortex', 'fimbria']):
    # Get data for this region
    region_data = grouped_df[grouped_df['region_group'] == region]
    total_halos = len(region_data)
    
    # Calculate counts for each category
    zero_halos = (region_data['core_vols_um'] == 0).sum()
    small_halos = ((region_data['core_vols_um'] > 0) & (region_data['core_vols_um'] <= 1000)).sum()
    large_halos = (region_data['core_vols_um'] > 1000).sum()
    
    # Create sizes and labels
    sizes = [zero_halos, small_halos, large_halos]
    labels = ['No core', 'Small core', 'Large core']
    
    # Calculate percentages for labels
    percentages = [count/total_halos * 100 for count in sizes]
    labels = [f'{l}\n({p:.1f}%)' for l, p in zip(labels, percentages)]
    
    # Plot pie chart
    wedges, texts, autotexts = ax.pie(sizes, 
                                     labels=labels,
                                     colors=colors,
                                     autopct='',
                                     startangle=90)
    
    # Title
    ax.set_title(f'{region}', fontsize=label_fontsize)

plt.tight_layout()
plt.savefig(sav_fold + "core_volume_distribution_pies.svg", dpi=300, format='svg')
plt.show()


# === Compute total number of cores per region ===

# Define regions of interest
regions_of_interest = ['Isocortex', 'fimbria']

for region in regions_of_interest:
    region_data = grouped_df[grouped_df['region_group'] == region]
    total_halos = len(region_data)
    total_cores = (region_data['core_vols_um'] > 0).sum()  # count only halos with cores
    percent_with_cores = (total_cores / total_halos) * 100
    
    print(f"\n{region}:")
    print(f"  Total halos: {total_halos}")
    print(f"  Total cores: {total_cores}")
    print(f"  Percent with cores: {percent_with_cores:.2f}%")



#%% Make a GLOBAL plot of myelin diff in intensity!!! --- i.e. compare ALL regions and save in keys_df

# Prepare columns for results
keys_df = main_keys.copy(deep=True)
from tqdm import tqdm

# Add result columns
keys_df['mean_myelin_diff'] = np.nan
keys_df['mean_control_myelin'] = np.nan
keys_df['mean_5xFAD_myelin'] = np.nan
keys_df['n_halos_in_region'] = 0  # default to 0
keys_df['mean_core_vol'] = np.nan  # new column

keys_df['cuprizone_myelin'] = np.nan
keys_df['mean_cup_diff'] = np.nan

# Progress bar
tqdm.pandas()

# for idx, row in tqdm(keys_df.iterrows(), total=keys_df.shape[0], desc='Processing regions'):
#     region_name = row['names']

#     region_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name=region_name)

#     if not region_ids:
#         continue

#     region_ids = keys_df.iloc[region_ids]['ids'].tolist()
#     region_halos = coords_df[coords_df['region_ids'].isin(region_ids)]

#     n = len(region_halos)

#     # Store count and stats
#     keys_df.at[idx, 'n_halos_in_region'] = n
#     keys_df.at[idx, 'mean_myelin_diff'] = (region_halos['myelin_int'] - region_halos[control_col]).mean()
#     keys_df.at[idx, 'mean_control_myelin'] = region_halos[control_col].mean()
#     keys_df.at[idx, 'mean_5xFAD_myelin'] = region_halos['myelin_int'].mean()
#     keys_df.at[idx, 'mean_core_vol'] = region_halos['core_vols_um'].mean()
    
#     # keys_df.at[idx, 'cuprizone_myelin'] = region_halos['M266_MoE_CUPRIZONE_6wks_fused_myelin'].mean()
#     # keys_df.at[idx, 'mean_cup_diff'] = (region_halos['M266_MoE_CUPRIZONE_6wks_fused_myelin'] - region_halos['M256_P60_MoE_high_AAVs_3chs_fused_myelin']).mean()



import warnings
warnings.filterwarnings('ignore')
from scipy.stats import sem  # standard error of the mean

for idx, row in tqdm(keys_df.iterrows(), total=keys_df.shape[0], desc='Processing regions'):
    region_name = row['names']

    region_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name=region_name)
    if not region_ids:
        continue

    region_ids = keys_df.iloc[region_ids]['ids'].tolist()
    region_halos = coords_df[coords_df['region_ids'].isin(region_ids)]

    if region_halos.empty:
        continue

    # Group by experiment
    grouped = region_halos.groupby('exp')

    # Myelin diff (control - 5xFAD) per experiment, avoid deprecation warning
    exp_myelin_diff = grouped[['myelin_int', control_col]].apply(
        lambda g: np.nanmean(g['myelin_int'] - g[control_col])
    )

    # Other per-experiment means
    exp_control = grouped[control_col].mean()
    exp_fad = grouped['myelin_int'].mean()
    exp_core = grouped['core_vols_um'].mean()
    exp_counts = grouped.size()

    # Final means across experiments
    keys_df.at[idx, 'n_halos_in_region'] = np.nanmean(exp_counts)
    keys_df.at[idx, 'mean_myelin_diff'] = np.nanmean(exp_myelin_diff)
    keys_df.at[idx, 'sem_myelin_diff'] = sem(exp_myelin_diff, nan_policy='omit')
    keys_df.at[idx, 'mean_control_myelin'] = np.nanmean(exp_control)
    keys_df.at[idx, 'mean_5xFAD_myelin'] = np.nanmean(exp_fad)
    keys_df.at[idx, 'mean_core_vol'] = np.nanmean(exp_core)

    # Cuprizone-related values (across all halos)
    keys_df.at[idx, 'cuprizone_myelin'] = np.nanmean(
        region_halos['M266_MoE_CUPRIZONE_6wks_fused_myelin']
    )

    keys_df.at[idx, 'mean_cup_diff'] = np.nanmean(
        region_halos['M266_MoE_CUPRIZONE_6wks_fused_myelin'] -
        region_halos['M256_P60_MoE_high_AAVs_3chs_fused_myelin']
    )
    
    
    
    
    
####################### MAKE SURE TO EXCLUDE REGIONS IN KEYS_DF   ### ALSO NEED TO DO THIS GENERALLY???

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
    




from adjustText import adjust_text

#%% Also try comparing to core volume
# Prepare filtered data
plot_df = keys_df.copy()
plot_df = plot_df[plot_df['n_halos_in_region'] >= 30]
exclusions = ['olfactory', 'dorsal limb', 'optic nerve', 'optic tract']
mask = plot_df['names'].str.lower().str.contains('|'.join(exclusions))
plot_df = plot_df[~mask]
plot_df = plot_df.dropna(subset=['mean_core_vol', 'mean_myelin_diff'])

# Flip myelin_diff direction (more negative = more loss)
# plot_df['mean_myelin_diff'] *= -1

# Step 1: Get top 10 regions by myelin loss
top_loss = plot_df.nsmallest(10, 'mean_myelin_diff')

# Step 2: Ensure fimbria (fi) is in top_loss
fimbria_row = plot_df[plot_df['acronym'] == 'fi']
if not fimbria_row.empty and fimbria_row.index[0] not in top_loss.index:
    top_loss = pd.concat([top_loss, fimbria_row])

# Step 3: Get top 10 regions by core volume
top_core = plot_df.nlargest(10, 'mean_core_vol')

# Step 4: Combine and deduplicate
plot_df = pd.concat([top_loss, top_core]).drop_duplicates(subset='acronym').copy()

# Step 5: Assign categories
plot_df['category'] = 'Other'
plot_df.loc[top_loss.index, 'category'] = 'Top Myelin Loss'
plot_df.loc[top_core.index, 'category'] = 'Top Core Volume'



# Handle overlaps: keep only unique regions
plot_df = plot_df[plot_df['category'] != 'Other']

# Set color palette manually
palette = {
    'Top Myelin Loss': '#E754A6',  # Bright magenta
    'Top Core Volume': '#555555'   # Neutral dark gray
}

# === Plot
fig, ax = plt.subplots(figsize=(4, 3))
sns.scatterplot(
    data=plot_df,
    x='mean_core_vol',
    y='mean_myelin_diff',
    hue='category',
    palette=palette,
    s=35,
    marker='o',
    alpha=0.9,
    ax=ax
)

all_texts = []
for _, row in plot_df.iterrows():
    text = ax.text(
        row['mean_core_vol'],
        row['mean_myelin_diff'],
        row['acronym'],
        fontsize=10,
        ha='left',
        va='bottom',
        color='black'
    )
    all_texts.append(text)

# Adjust positions to avoid overlap
adjust_text(
    all_texts,
    only_move='xy',
    arrowprops=dict(arrowstyle='->', color='red'),
    ax=ax
)
# Format plot
ax.axhline(0, linestyle='--', linewidth=1.4, color='#E754A6', alpha=0.9)

ax.set_xlabel('Mean Core Volume (μm³)', fontsize=label_fontsize)
ax.set_ylabel('Myelin Intensity Loss\n(5xFAD - Control)', fontsize=label_fontsize)
ax.tick_params(axis='x', labelsize=label_fontsize)
ax.tick_params(axis='y', labelsize=tick_fontsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Optional y-axis limit
ax.set_ylim(bottom=-600, top=100)

# Clean legend
ax.legend(title=None, fontsize=label_fontsize, frameon=False, loc='best')

plt.tight_layout()
plt.savefig(sav_fold + "core_volume_vs_myelin_loss_highlighted.svg", dpi=300, format='svg')
plt.show()



 
    
    

#%% Utility functions for plotting
def prepare_plot_data(keys_df, min_halos=30, exclude_regions=None):
    """Prepare data for plotting by filtering and cleaning."""
    plot_df = keys_df.copy()
    plot_df = plot_df[plot_df['n_halos_in_region'] >= min_halos]
    
    if exclude_regions:
        mask = plot_df['names'].str.lower().str.contains('|'.join(exclude_regions))
        plot_df = plot_df[~mask]
    
    return plot_df

def get_region_colors(plot_df, keys_df):
    """Assign colors to regions based on whether they're in isocortex or fiber tracts."""
    colors = pd.Series('#8E8E8E', index=plot_df.index)  # Default color - neutral grey
    
    # Get isocortex regions
    isocortex_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Isocortex')
    isocortex_regions = keys_df.iloc[isocortex_ids]['ids'].tolist()
    
    # Get fiber tract regions
    fiber_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='fiber tracts')
    fiber_regions = keys_df.iloc[fiber_ids]['ids'].tolist()
    
    # Assign colors - using orange for isocortex
    colors[plot_df['ids'].isin(isocortex_regions)] = '#FFA500'  # Orange for isocortex
    colors[plot_df['ids'].isin(fiber_regions)] = '#27AE60'      # Emerald green for fiber tracts
    
    return colors
    

    
    
    

def plot_brain_regions(plot_df, keys_df, x_col, y_col, title, sav_path, 
                      region_filter=None, calculate_difference=False, 
                      diff_cols=None, single_color=None, min_halos=30,
                      n_labels=10, label_specific_regions=None,
                      ylim=(-600, 100), xlim=None, highlight_max=False,
                      label_filter_region=None):
    """
    Unified function for plotting brain region comparisons.
    
    Additional Parameters:
    -----------
    label_filter_region : str, optional
        If provided, only label regions from this region type (e.g., 'fiber tracts')
    """
    # Copy and filter data
    plot_df = plot_df.copy()
    
    # Apply region filter if specified
    if region_filter:
        region_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name=region_filter)
        plot_df = plot_df.iloc[region_ids]
    
    # Filter by minimum halos
    plot_df = plot_df[plot_df['n_halos_in_region'] >= min_halos]
    
    # Calculate difference if requested
    if calculate_difference:
        if not diff_cols or len(diff_cols) != 2:
            raise ValueError("Must provide two column names for difference calculation")
        col1, col2 = diff_cols
        plot_df[y_col] = plot_df[col1] - plot_df[col2]
    
    # Drop rows with missing data
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(4, 3))
    
    if single_color:
        # Single color plot
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            color=single_color,
            s=20,
            marker='o',
            alpha=0.8,
            ax=ax
        )
    else:
        # Region-colored plot
        colors = get_region_colors(plot_df, keys_df)
        for color in colors.unique():
            mask = colors == color
            sns.scatterplot(
                data=plot_df[mask],
                x=x_col,
                y=y_col,
                color=color,
                s=20,
                marker='o',
                alpha=0.8,
                ax=ax
            )
    
    # Add horizontal reference line
    ax.axhline(0, linestyle='--', linewidth=1.4, color='#E754A6', alpha=0.9)
    
    # Filter regions for labeling if requested
    if label_filter_region:
        filter_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name=label_filter_region)
        filter_regions = keys_df.iloc[filter_ids]['ids'].tolist()
        plot_df_filtered = plot_df[plot_df['ids'].isin(filter_regions)]
    else:
        plot_df_filtered = plot_df
    
    # Get regions to label
    if highlight_max:
        top_regions = plot_df_filtered.nlargest(n_labels, y_col)
    else:
        top_regions = plot_df_filtered.nsmallest(n_labels, y_col)
    regions_to_label = top_regions.copy()
    
    # Add specific regions if requested
    if label_specific_regions:
        for region in label_specific_regions:
            region_row = plot_df[plot_df['acronym'] == region]
            if not region_row.empty and region_row.index[0] not in regions_to_label.index:
                regions_to_label = pd.concat([regions_to_label, region_row])
    
    # Add labels with adjustText
    all_texts = []
    for _, row in regions_to_label.iterrows():
        text = ax.text(
            row[x_col],
            row[y_col],
            row['acronym'],
            fontsize=10,
            ha='left',
            va='bottom',
            color='black'
        )
        all_texts.append(text)
    
    adjust_text(
        all_texts,
        only_move='xy',
        arrowprops=dict(arrowstyle='->', color='red'),
        ax=ax
    )
    
    # Format axes
    ax.set_xlabel('Mean Control Myelin Intensity', fontsize=label_fontsize)
    ax.set_ylabel(title, fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set axis limits
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if xlim:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    
    plt.tight_layout()
    plt.savefig(sav_path, dpi=300, format='svg')
    plt.show()

#%% Plot myelin intensity diff vs. control intensity
# All regions
plot_brain_regions(
    plot_df=keys_df,
    keys_df=keys_df,
    x_col='mean_control_myelin',
    y_col='mean_myelin_diff',
    title='Myelin Intensity Loss\n(5xFAD - Control)',
    sav_path=sav_fold + "myelin_loss_vs_control.svg",
    label_specific_regions=['fi'],
    label_filter_region='fiber tracts'  # Only label fiber tract regions
)

#%% Plot control vs. CUPRIZONE myelin intensity
plot_brain_regions(
    plot_df=keys_df,
    keys_df=keys_df,
    x_col='mean_control_myelin',
    y_col='mean_cup_diff',
    title='Myelin Intensity Loss\n(Cuprizone - Control)',
    sav_path=sav_fold + "CUPRIZONE_loss_vs_control.svg",
    label_specific_regions=['fi'],
    label_filter_region='fiber tracts'  # Only label fiber tract regions
)

#%% Plot difference between cuprizone and 5xFAD effects
# All regions
plot_brain_regions(
    plot_df=keys_df,
    keys_df=keys_df,
    x_col='mean_control_myelin',
    y_col='diff_between_cup_and_5xFAD',
    title='Difference in Myelin Loss\n(Cuprizone - 5xFAD)',
    sav_path=sav_fold + "diff_between_cup_and_5xFAD_loss_vs_control.svg",
    calculate_difference=True,
    diff_cols=('mean_cup_diff', 'mean_myelin_diff'),
    label_specific_regions=['fi'],
    highlight_max=True,  # Highlight regions with highest difference
    ylim=(-300, 300),  # Updated y-axis limits
    label_filter_region='fiber tracts'  # Only label fiber tract regions
)





#%% OPTIONAL --- For halos found in fimbria and cortex and alveus, get random sample of them and get a crop of 128 x 128 x 32
# get also plot the coords of the crop to get binary outline, as well as myelin image


    
# # For re-running code, make sure to drop region group
# if 'region_group' in coords_df.columns:
#     coords_df = coords_df.drop(columns='region_group')

# # --- 1. Define your new regions of interest ---
# regions_of_interest = ['fimbria', 'Isocortex', 'alveus']

# # --- 2. Build a full dictionary mapping ---
# region_id_dict = {}

# for region in regions_of_interest:
#     subregion_idx = get_sub_regions_atlas(main_keys, child_id=[], sub_keys=[], reg_name=region)
#     subregion_ids = main_keys.iloc[subregion_idx]['ids'].tolist()
#     region_id_dict[region] = subregion_ids

# # --- 3. Now label halos based on region IDs ---
# def assign_region(rid):
#     for region, id_list in region_id_dict.items():
#         if rid in id_list:
#             return region
#     return None  # if it doesn't belong to any

# coords_df['region_name'] = coords_df['region_ids'].apply(assign_region)


# crop_shape = np.array([32, 128, 128])  # (z,y,x) size


# crops = []
# avg_crops = []


# for region in regions_of_interest:
    
        
#     halo_counter = 0
#     core_counter = 0


#     avg_halo_sum = np.zeros(crop_shape, dtype=float)  # non-shifted halo masks
#     avg_core_sum = np.zeros(crop_shape, dtype=float)  # shifted core masks
    
    
    
#     # Filter halos in this region
#     region_df = coords_df[coords_df['region_name'] == region]
    
#     # Sample up to 100 halos
#     sampled_df = region_df.sample(n=min(300, len(region_df)), random_state=88)
    
#     for idx, halo in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc=f"Processing {region} halos"):
        
#         ### 1. Get centroid corrected by offset
#         centroid = np.array(halo['centroid'])  # (x,y,z)
#         s_c = np.array(halo['xyz_offset'])     # (z,y,x)

#         # Correct centroid to (z,y,x) order and add offset
#         centroid_corrected = np.array([
#             centroid[0] + s_c[2],  # z
#             centroid[1] + s_c[1],  # y
#             centroid[2] + s_c[0],  # x
#         ])
        
        

#         # Define start and end of crop
#         start = np.round(centroid_corrected - crop_shape // 2).astype(int)
#         end = start + crop_shape

#         start_z, start_y, start_x = start
#         end_z, end_y, end_x = end

#         ### 2. Crop the datasets
#         crop_myelin = dset[start_z:end_z, start_y:end_y, start_x:end_x]
#         crop_halo = dset_halo[start_z:end_z, start_y:end_y, start_x:end_x]
#         crop_core = dset_congo[start_z:end_z, start_y:end_y, start_x:end_x]

#         # Crop the control volume separately based on bbox_dset_c1
#         bbox_c1 = np.array(halo['bbox_dset_c1'])
#         start_z_c1, start_y_c1, start_x_c1 = bbox_c1[0], bbox_c1[1], bbox_c1[2]
#         end_z_c1 = start_z_c1 + crop_shape[0]
#         end_y_c1 = start_y_c1 + crop_shape[1]
#         end_x_c1 = start_x_c1 + crop_shape[2]

#         crop_control = dset_c1[start_z_c1:end_z_c1, start_y_c1:end_y_c1, start_x_c1:end_x_c1]

#         ### 3. Build halo and core masks
#         coords_halo = np.copy(halo['coords_halo'])  # (N,3)
#         coords_core = np.copy(halo['coords_core'])  # (M,3), could be empty
        
        
#         # 1. Correct halo coords by adding offset
#         coords_halo_corrected = np.stack([
#             coords_halo[:,0] + s_c[2],  # z
#             coords_halo[:,1] + s_c[1],  # y
#             coords_halo[:,2] + s_c[0],  # x
#         ], axis=1)
        
#         # 2. Same for core coords
#         coords_core_corrected = np.stack([
#             coords_core[:,0] + s_c[2],  # z
#             coords_core[:,1] + s_c[1],  # y
#             coords_core[:,2] + s_c[0],  # x
#         ], axis=1) if len(coords_core) > 0 else np.empty((0,3))
        
#         # 3. Now adjust coords relative to crop start
#         coords_halo_adj = coords_halo_corrected - start  # start = (start_z, start_y, start_x)
#         coords_core_adj = coords_core_corrected - start
        
#         # 4. Only keep points inside the crop
#         valid_halo = np.all((coords_halo_adj >= 0) & (coords_halo_adj < crop_shape), axis=1)
#         coords_halo_adj = coords_halo_adj[valid_halo].astype(int)
        
#         valid_core = np.all((coords_core_adj >= 0) & (coords_core_adj < crop_shape), axis=1)
#         coords_core_adj = coords_core_adj[valid_core].astype(int)
        
#         # 5. Build masks
#         halo_mask = np.zeros(crop_shape, dtype=bool)
#         core_mask = np.zeros(crop_shape, dtype=bool)
        
#         halo_mask[coords_halo_adj[:,0], coords_halo_adj[:,1], coords_halo_adj[:,2]] = True
#         if len(coords_core_adj) > 0:
#             core_mask[coords_core_adj[:,0], coords_core_adj[:,1], coords_core_adj[:,2]] = True

#         ### 4. Store result
#         crops.append({
#             'region': region,
#             'halo_id': idx,
#             'myelin_crop': crop_myelin,
#             'halo_crop': crop_halo,
#             'core_crop': crop_core,
#             'control_crop': crop_control,
#             'halo_mask': halo_mask,
#             'core_mask': core_mask,
#         })
                    

#         halo_stack = np.stack([
#             crop_myelin.astype(np.uint16),   # Myelin raw
#             crop_halo.astype(np.uint16),     # Halo raw
#             crop_core.astype(np.uint16),     # Core raw
#             halo_mask.astype(np.uint16) * 65535,     # Halo mask
#             core_mask.astype(np.uint16) * 65535,     # Core mask
#             crop_control.astype(np.uint16),  # Control raw
#         ], axis=0)  
                    
#         # Add t=1 dimension at the front to match (t, c, z, y, x)
#         halo_stack = halo_stack[None, :, :, :, :] 
#         halo_stack = np.transpose(halo_stack, (0,2,1,3,4))

        
#         # Save as multi-page TIFF
#         if idx <= 20:  ### only plot first 20
#             tiff.imwrite(sav_dir + region + '_' + str(idx) + '_example_myelin_halo_core_composite', halo_stack, imagej=True,
#                          metadata={'axes': 'TZCYX'})

#             middle_slice_stack = halo_stack[0, halo_stack.shape[1] // 2, :, :, :]  # shape (C,Y;;;;;;;;;e,X)
#             middle_slice_stack = np.expand_dims(middle_slice_stack, axis=0)
#             middle_slice_stack = np.expand_dims(middle_slice_stack, axis=0)
#             tiff.imwrite(sav_dir + region + '_' + str(idx) + '_MIDSLICE_composite', middle_slice_stack, imagej=True,
#                          metadata={'axes': 'TZCYX'})


#         # --- Accumulate halo crops directly ---
#         avg_halo_sum += crop_halo.astype(float)
#         halo_counter += 1
        
              
                
#         if np.any(coords_core):
        
#             core_object_centroid = coords_core_corrected.mean(axis=0)  # (z,y,x)
        
#             # Centered crop window
#             start = np.round(core_object_centroid - crop_shape // 2).astype(int)
#             end = start + crop_shape
        
#             start_z, start_y, start_x = start
#             end_z, end_y, end_x = end
        
#             # Safely crop (you could add boundary checks if needed)
#             recentered_core_crop = dset_congo[start_z:end_z, start_y:end_y, start_x:end_x]
        
#             # Accumulate
#             avg_core_sum += recentered_core_crop.astype(float)
#             core_counter += 1


        
       
#     ### Get average shape of halo and core masks
#     avg_halo_sum = avg_halo_sum / halo_counter
    
#     if core_counter > 0:
#         avg_core_sum = avg_core_sum / core_counter
#     else:
#         avg_core_sum = np.zeros(crop_shape, dtype=float)                        
             
#     avg_crops.append({
#         'region': region,
#         'avg_halo': avg_halo_sum,
#         'avg_core': avg_core_sum,
#     })
        
    
    
# for crop in avg_crops:
#     region = crop['region']
#     avg_halo = crop['avg_halo']
#     avg_core = crop['avg_core']
    
#     plot_max(avg_halo, vmin=200, vmax=350)
#     plt.title(f"{region} - Avg Halo")
#     plt.axis('off')

#     plot_max(avg_core, vmin=120, vmax=350)
#     plt.title(f"{region} - Avg Core")
#     plt.axis('off')

#     plt.tight_layout()

#     tiff.imwrite(sav_dir + region + '_avg_halo.tif', np.uint16(avg_halo))
#     tiff.imwrite(sav_dir + region + '_avg_core.tif', np.uint16(avg_core))






    

    

   


    

    

    
    
    
