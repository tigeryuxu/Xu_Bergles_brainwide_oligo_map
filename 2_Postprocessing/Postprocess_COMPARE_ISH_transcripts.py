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



sav_fold = '/media/user/4TB_SSD/Plot_outputs/'


pad = True

XY_res = 1.152035240378141
Z_res = 5
res_diff = XY_res/Z_res

ANTS = 1



#%%%% Parse the json file so we can choose what we want to extract or mask out
reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'

ref_atlas = tiff.imread(reference_atlas)
ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)


cc_allen = regionprops(ref_atlas, cache=False)
cc_labs_allen = [region['label'] for region in cc_allen]
cc_labs_allen = np.asarray(cc_labs_allen)

#%% Keys
with open('../atlas_ids/atlas_ids.json') as json_file:
    data = json.load(json_file)
 
     
data = data['msg'][0]
 
 
print('Extracting main key volumes')
keys_dict = get_ids_all(data, all_keys=[], keywords=[''])  
main_keys = pd.DataFrame.from_dict(keys_dict)


keys_tmp = get_ids_all(data, all_keys=[], keywords=[''])  
keys_tmp = pd.DataFrame.from_dict(keys_tmp)



#%%%% Parse pickles from all folders --- Jacob comment out all below here until pickle loading
""" Loop through all the folders and pool the dataframes!!!"""

all_coords_df = []
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
    
    hindbrain_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Hindbrain')  ### JACOB
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


#%% Comment out everything before this and just load in pickle

import pickle as pkl
with open(sav_fold + 'df_concat_ISH', 'wb') as file:
    pkl.dump(df_concat, file)

# # Load pickle
# with open(sav_fold + 'df_concat_ISH', 'rb') as file:
#     df_concat = pkl.load(file)


#%%% Experiment name
plot_all = False


fontsize = 14

fig_stand = np.asarray([4.5, 5])
fig_long = np.asarray([4.8,4])

P60_x_lim = 18000




ish_density = '/media/user/4TB_SSD/spatial_ISH_probes/SagittalGeneExpressionDensityByAllenRegion_regionMetadata.pkl'
ish_energy = '/media/user/4TB_SSD/spatial_ISH_probes/SagittalGeneExpressionenergyByAllenRegion_regionMetadata.pkl'
conversion_pickle = '/media/user/4TB_SSD/spatial_ISH_probes/AllenISH_Sagittal_GeneInfo.pkl'



import pickle as pkl
with open(ish_density, 'rb') as file:
    ish_density = pkl.load(file)

with open(ish_energy, 'rb') as file:
    ish_energy = pkl.load(file)


with open(conversion_pickle, 'rb') as file:
    conversion = pkl.load(file)




exp_name = 'pooled ISH'



### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
pooled_df = df_concat[df_concat['exp'].isin(['P60', 'P60_F'])]
df_means = pooled_df.groupby(pooled_df.index).mean(numeric_only=True)
df_means['acronym'] = keys_df['acronym']
df_means['dataset'] = keys_df['dataset']
df_means['names'] = keys_df['names']
df_means['children'] = keys_df['children']



#%% First clean up ish data and make sure no duplicate header names
string_cols = ish_energy[['atlas_id', 'acronym', 'id', 'safe_name', 'ontology_id', 'index']]

numeric_cols = ish_energy.drop(columns=['atlas_id', 'acronym', 'id', 'safe_name', 'ontology_id', 'index'])

# Step 2: Average duplicate numeric columns
numeric_avg = numeric_cols.groupby(axis=1, level=0).mean()

# Step 3: Combine back with the string columns
df_result = pd.concat([string_cols, numeric_avg], axis=1)


### Also drop duplicates from conversion table    
conversion = conversion.drop_duplicates(subset='entrez')
conversion = conversion.drop_duplicates(subset='gene_name')




#%% Concatenate both dataframes (ISH and OL density)
# Step 1: Convert both to DataFrames with the value as index
df1_indexed = df_result.set_index('id')
df2_indexed = df_means.set_index('ids')

# Step 2: Outer join on the index (the integer value)
result = pd.merge(df1_indexed, df2_indexed, left_index=True, right_index=True, how='outer')

# Step 3: Reset index if you want the integer back as a column
result = result.reset_index()

result = result.dropna(axis=1, how='all')  ### drop all nan columns


zzz

#%% Compute correlations rapidly
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


def corr_analysis(df, target_col):
    # Assume 'cell_density' is the target column
    target = df['density_W']
    # X = df.drop(columns='cell_density')
    
    X = df[[col for col in df.columns if not isinstance(col, str)]]  ### Drop all columns that are string, leaving only genes
    
    genes = X.columns
    
    
    correlations = []
    p_values = []
    
    for i, gene in enumerate(genes):
        x = X[gene]
        valid = target.notna() & x.notna()
        if valid.sum() > 2:
            x_valid = x[valid]
            y_valid = target[valid]
            
            # Center and normalize
            x_c = x_valid - x_valid.mean()
            y_c = y_valid - y_valid.mean()
            
            r = np.dot(x_c, y_c) / ((len(x_c) - 1) * x_c.std(ddof=1) * y_c.std(ddof=1))
            r = np.clip(r, -1, 1)
    
            # p-value
            t_stat = r * np.sqrt((len(x_c) - 2) / (1 - r**2))
            p = 2 * t.sf(np.abs(t_stat), df=len(x_c) - 2)
    
            correlations.append(r)
            p_values.append(p)
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)
    
    # Build result
    cor_df = pd.DataFrame({
        'gene': genes,
        'correlation': correlations,
        'p_value': p_values
    })
    
    # Adjust p-values
    cor_df['p_adj'] = multipletests(cor_df['p_value'].fillna(1), method='fdr_bh')[1]
    
    
    return cor_df


cor_df = corr_analysis(result, target_col='density_W')

#%% Also concatenate gene name
cor_df.rename(columns={'gene': 'entrez'}, inplace=True)
df_merged = cor_df.merge(conversion, on='entrez', how='left')  ### left keeps all values from result df
df_merged = df_merged.sort_values(by='correlation', ascending=False)

# because conversion was altered earlier to look for duplicates, remove any misaligned (gene_name = nan) entries
df_merged = df_merged.dropna(subset=['gene_name'])

import pickle as pkl
with open(sav_fold + 'correlation_ISH_OL_density', 'wb') as file:
    pkl.dump(df_merged, file)





df_merged.loc[df_merged['gene_name'] == 'Ncam1']
df_merged.loc[df_merged['gene_name'] == 'Fgf18']





#%% Plot 3D visualization of correlation map

# First get entrez ID corresponding to gene name
def get_3D_gene_map(result, df_merged, goi, scale=10000):
    entrez_id = df_merged.loc[df_merged['gene_name'] == goi]['entrez']
    
    # Then get column corresponding to density of gene
    col_ref = result[['acronym_x', 'safe_name', 'level_0']]
    gene_expression = result[entrez_id]
    
    gene_expression = pd.concat([col_ref, gene_expression], axis=1)
    
    # gene_expression = gene_expression.dropna()
    
    """ Add to Allen map """
    gene_map_3D = np.zeros(np.shape(ref_atlas))
    for idx, row in gene_expression.iterrows():
        
        id_g = row['level_0']
        value = row[entrez_id]  * scale ### SCALE UP BECAUSE VALUE CANT BE REPRESENTED IN DECIMALS
        cur_id = np.where(cc_labs_allen == id_g)[0]
        
        #print(cur_id)
        if len(cur_id) == 0:  ### if it does not exists in atlas
            continue
        
        
        # if value is nan, replace with very small number so doesn't zero out
        if pd.isna(value).any():
            value = 0
        
        
        cur_coords = cc_allen[cur_id[0]]['coords']
        gene_map_3D[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = value
       
        
    return gene_map_3D



goi = 'Mog'
gene_map_3D = get_3D_gene_map(result, df_merged, goi, scale=1)
plt.figure(); plt.imshow(gene_map_3D[300])

tiff.imwrite(sav_fold + 'ISH_' + goi + '.tif', gene_map_3D)

goi = 'Bai1'
gene_map_3D = get_3D_gene_map(result, df_merged, goi, scale=1)
plt.figure(); plt.imshow(gene_map_3D[300])
tiff.imwrite(sav_fold + 'ISH_' + goi + '.tif', gene_map_3D)


goi = 'Mobp'
gene_map_3D = get_3D_gene_map(result, df_merged, goi, scale=1)
plt.figure(); plt.imshow(gene_map_3D[300])
tiff.imwrite(sav_fold + 'ISH_' + goi + '.tif', gene_map_3D)

# import napari
# viewer = napari.Viewer()
# viewer.add_image(gene_map_3D)
# viewer.show(block=True)


#%%## Make an OL density map

""" Add to Allen map """
OL_density_by_region = np.zeros(np.shape(ref_atlas))
for idx, row in result.iterrows():
    
    id_g = row['level_0']
    value = row['density_W']  #* scale ### SCALE UP BECAUSE VALUE CANT BE REPRESENTED IN DECIMALS
    cur_id = np.where(cc_labs_allen == id_g)[0]
    
    #print(cur_id)
    if len(cur_id) == 0:  ### if it does not exists in atlas
        continue
    
    if np.isnan(value):
        value = 1
    
    cur_coords = cc_allen[cur_id[0]]['coords']
    OL_density_by_region[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = value
   
plt.figure(); plt.imshow(OL_density_by_region[300])
tiff.imwrite(sav_fold + 'ISH_matched_OL_density.tif', OL_density_by_region)


OL_density_by_region[gene_map_3D <= 1] = 0
plt.figure(); plt.imshow(OL_density_by_region[300])
tiff.imwrite(sav_fold + 'ISH_matched_OL_density_MASKED.tif', OL_density_by_region)



#%% Make a cuprizone density map

cup_df = df_concat[df_concat['exp'].isin(['Cuprizone'])]
df_means_cup = cup_df.groupby(cup_df.index).mean(numeric_only=True)
df_means_cup['acronym'] = keys_df['acronym']
df_means_cup['dataset'] = keys_df['dataset']
df_means_cup['names'] = keys_df['names']
df_means_cup['children'] = keys_df['children']


""" Add to Allen map """
cup_density = np.zeros(np.shape(ref_atlas))
for idx, row in cup_df.iterrows():
    
    id_g = row['ids']
    value = row['density_W']  #* scale ### SCALE UP BECAUSE VALUE CANT BE REPRESENTED IN DECIMALS
    cur_id = np.where(cc_labs_allen == id_g)[0]
    
    #print(cur_id)
    if len(cur_id) == 0:  ### if it does not exists in atlas
        continue
    
    if np.isnan(value):
        value = 1
    
    cur_coords = cc_allen[cur_id[0]]['coords']
    cup_density[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = value
   
plt.figure(); plt.imshow(cup_density[300])
tiff.imwrite(sav_fold + 'ISH_matched_OL_density_CUP.tif', cup_density)


cup_density[gene_map_3D <= 1] = 0
plt.figure(); plt.imshow(cup_density[300])
tiff.imwrite(sav_fold + 'ISH_matched_OL_density_MASKED_CUP.tif', cup_density)



### Make a cuprizone FOLD CHANGE map
# cup_fc = (OL_density_by_region - cup_density)/OL_density_by_region
# cup_fc = (cup_density - OL_density_by_region)/OL_density_by_region
cup_fc = (cup_density)/OL_density_by_region
cup_fc = np.nan_to_num(cup_fc)
plt.figure(); plt.imshow(cup_fc[300])
tiff.imwrite(sav_fold + 'ISH_matched_OL_density_MASKED_CUP_FC.tif', cup_fc)

zzz


#%% Subregion-specific correlations
isocortex_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Isocortex')
isocortex_atlas_ids = keys_df.iloc[isocortex_idx]

# run corr analysis















#%% Plot laminar profile 
    
    
import re
from scipy.stats import zscore    
def parse_layer_and_macroregion(name):
    if pd.isna(name):
        return np.nan, np.nan
    match = re.search(r'(.*?)(?:,\s*|\s+)layer\s*(\d(?:\/\d)?[a-z]?)$', name, re.IGNORECASE)
    if match:
        macro = match.group(1).strip()
        layer = 'L' + match.group(2).strip().lower()
        return macro, layer
    else:
        return np.nan, np.nan

def preprocess_cortex_layers(df, names_col='safe_name'):
    df[['macroregion', 'layer']] = df[names_col].apply(lambda x: pd.Series(parse_layer_and_macroregion(x)))
    df = df.dropna(subset=['macroregion', 'layer'])

    # Standardize and filter known cortical layers
    layer_order = ['L1', 'L2/3', 'L4', 'L5', 'L6a', 'L6b']
    df = df[df['layer'].isin(layer_order)]
    df['layer'] = pd.Categorical(df['layer'], categories=layer_order, ordered=True)

    return df

def compute_layer_profiles(df, group_cols=['macroregion', 'layer']):
    return df.groupby(group_cols).mean(numeric_only=True).reset_index()


def safe_zscore(series):
    non_nan = series.dropna()
    if len(non_nan) < 2:
        return pd.Series([np.nan] * len(series), index=series.index)
    z = (non_nan - non_nan.mean()) / non_nan.std(ddof=0)
    return z.reindex(series.index)  # preserves NaNs in original positions



def plot_laminar_profile(grouped_df, region_name, genes=None, density_col='density_W', zscore_data=True, interpolate_for_plot=True):
    """
    Plots the laminar profile of OL density and gene expression for a given macroregion.
    Optionally applies z-scoring across layers and interpolates missing values for plotting.
    """
    df = grouped_df[grouped_df['macroregion'] == region_name].sort_values('layer').copy()

    # Apply NaN-safe z-scoring
    if zscore_data:
        if density_col in df.columns:
            df[density_col + '_z'] = safe_zscore(df[density_col])
        if genes:
            for gene in genes:
                if gene in df.columns:
                    df[gene + '_z'] = safe_zscore(df[gene])

    plt.figure(figsize=(6, 4))

    # Interpolate for plotting continuity
    plot_df = df.copy()
    if interpolate_for_plot:
        for col in plot_df.columns:
            if col.endswith('_z') or col == density_col:
                plot_df[col] = plot_df[col].interpolate(method='linear', limit_direction='both')

    # Plot OL density
    if density_col in df.columns:
        y_col = density_col + '_z' if zscore_data else density_col
        plt.plot(plot_df['layer'], plot_df[y_col], label='OL Density', marker='o', linewidth=2)

    # Plot genes
    if genes:
        for gene in genes:
            if gene in df.columns:
                y_col = gene + '_z' if zscore_data else gene
                plt.plot(plot_df['layer'], plot_df[y_col], label=gene, marker='o')

    plt.title(f"Laminar Profile: {region_name} ({'Z-scored' if zscore_data else 'Raw'})")
    plt.xlabel("Layer")
    plt.ylabel("Z-score" if zscore_data else "Mean Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
    




# Step 1: Preprocess and extract laminar structure
cortex_df = preprocess_cortex_layers(result)



def map_entrez_to_gene_names(df, df_merged):
    """
    Renames columns in df (which uses float Entrez IDs) using gene names from df_merged.
    Keeps all other columns unchanged.
    """
    rename_dict = df_merged.set_index('entrez')['gene_name'].dropna().to_dict()

    # Convert to float to match column dtype
    rename_dict = {float(k): v for k, v in rename_dict.items()}

    return df.rename(columns=rename_dict)

cortex_df = map_entrez_to_gene_names(cortex_df, df_merged)

# Drop all remaining unmatched numerical columns --- happens because duplicates erased from df_merge
cortex_df = cortex_df[[col for col in cortex_df.columns if not isinstance(col, float)]]
# also drop columns that are irrelavant
cortex_df = cortex_df.drop(columns=[       'num_large_R', 'density_R', 'atlas_vol_W_relative', 'num_OLs_W',
       'num_large_W', 'age', 'density_LARGE_W',
       'num_large_W_CLEAN', 'density_LARGE_W_CLEAN',
       'num_large_L', 'density_L', 'atlas_vol_R_relative', 'num_OLs_R',
       'acronym_y', 
       
       ])

# Step 2: Compute mean values per [region, layer]
grouped_profiles = compute_layer_profiles(cortex_df)

# Step 3: Plot laminar profiles for specific regions and genes
# genes_to_plot = ['Fgf18', 'Ncam1']  # Or use float entrez IDs if that's your column format
genes_to_plot = ['Mog', 'Fgf18', 'Mag']
plot_laminar_profile(grouped_profiles, 'Primary somatosensory area mouth', genes=genes_to_plot)
plot_laminar_profile(grouped_profiles, 'Prelimbic area', genes=genes_to_plot)   
plot_laminar_profile(grouped_profiles, 'Perirhinal area', genes=genes_to_plot)    
plot_laminar_profile(grouped_profiles, 'Retrosplenial area dorsal part', genes=genes_to_plot)    
    



secreted_genes = [
    'Ngf', 'Bdnf', 'Ntf3', 'Ntf4',         # Neurotrophins
    'Fgf1', 'Fgf2', 'Fgf3', 'Vegfa', 'Pdgfa', 'Pdgfb',  # Growth Factors
    'Il6', 'Lif', 'Cntf',                  # Cytokines
    'Npy', 'Sst', 'Vip', 'Cartpt', 'Ghrh', 'Crh', 'Trh', 'Oxtr', 'Avp',  # Neuropeptides
    'Kal1', 'Hen1', 'Scg2', 'Chgb'         # Other secreted proteins
]
plot_laminar_profile(grouped_profiles, 'Primary somatosensory area mouth', genes=secreted_genes)    
    



#%% Now try finding top hits that trend same as OL density?

from scipy.stats import pearsonr

def find_genes_matching_ol_profile(grouped_df, region_name, density_col='density_W', min_layers=4, top_n=10, plot=True, compare_with=None, variance_threshold=0.0, zscore_plot=True):
    """
    Finds and optionally plots top genes whose laminar expression profile most closely matches OL density profile.
    Uses Pearson correlation; optionally z-scores profiles before plotting for better visual comparability.
    """
    df = grouped_df[grouped_df['macroregion'] == region_name].sort_values('layer').copy()
    df_valid = df.dropna(subset=[density_col])

    if df_valid.shape[0] < min_layers:
        print(f"Not enough valid layers in {region_name} to compute correlation.")
        return pd.DataFrame()

    density_profile = df_valid[density_col]
    gene_cols = [col for col in df.columns if isinstance(col, str) and col not in ['macroregion', 'layer', density_col]]

    corrs = []
    for gene in gene_cols:
        expr = df_valid[gene]
        valid = expr.notna() & density_profile.notna()
        if valid.sum() >= min_layers and expr[valid].var() > variance_threshold:
            r, p = pearsonr(expr[valid], density_profile[valid])
            corrs.append((gene, r, p))

    cor_df = pd.DataFrame(corrs, columns=['gene', 'correlation', 'p_value'])
    cor_df_full = cor_df.copy()
    cor_df = cor_df.sort_values(by='correlation', ascending=False).head(top_n)

    if plot and not cor_df.empty:
        plt.figure(figsize=(7, 5))
        region_df = df.set_index('layer')

        if zscore_plot:
            region_df_z = region_df.copy()
            region_df_z[density_col] = zscore(region_df[density_col].dropna())
            for gene in cor_df['gene']:
                if gene in region_df.columns:
                    region_df_z[gene] = zscore(region_df[gene].dropna())
            region_df = region_df_z

        for gene in cor_df['gene']:
            if gene in region_df.columns:
                plt.plot(region_df.index, region_df[gene], label=gene, marker='o')

        plt.plot(region_df.index, region_df[density_col], label='OL Density', color='black', linewidth=2, linestyle='--')

        if compare_with:
            df2 = grouped_df[grouped_df['macroregion'] == compare_with].sort_values('layer')
            df2 = df2.groupby('layer').mean(numeric_only=True)
            if not df2.empty:
                if zscore_plot:
                    df2[density_col] = zscore(df2[density_col].dropna())
                plt.plot(df2.index, df2[density_col], label=f'{compare_with} OL Density', color='gray', linewidth=2, linestyle=':')

        plt.title(f"Top {top_n} Genes Correlated with OL Density in {region_name}")
        plt.xlabel("Layer")
        plt.ylabel("Z-score" if zscore_plot else "Expression / Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return cor_df, cor_df_full



cor_df, cor_df_full = find_genes_matching_ol_profile(
    grouped_profiles,
    region_name='Primary somatosensory area mouth',
    compare_with='Prelimbic area',
    top_n=10,
    # variance_threshold=1e-6,
    zscore_plot=False
)





zzz




#%% Highlight regions of interest for injury and recovery!!! See if anything correlates in the cortex!!!

""" ALSO --- compare cuprizone injury and recovery regions of interest??? """




exp_name = 'cup ISH'



### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
cup_df = df_concat[df_concat['exp'].isin(['Cuprizone','Cuprizone_NEW'])]
df_means = cup_df.groupby(df_means.index).mean(numeric_only=True)
df_means['acronym'] = keys_df['acronym']
df_means['dataset'] = keys_df['dataset']
df_means['names'] = keys_df['names']
df_means['children'] = keys_df['children']


#%% Concatenate both dataframes (ISH and OL density)
# Step 1: Convert both to DataFrames with the value as index
df1_indexed = df_result.set_index('id')
df2_indexed = df_means.set_index('ids')

# Step 2: Outer join on the index (the integer value)
cup_result = pd.merge(df1_indexed, df2_indexed, left_index=True, right_index=True, how='outer')

# Step 3: Reset index if you want the integer back as a column
cup_result = cup_result.reset_index()

cup_result = cup_result.dropna(axis=1, how='all')  ### drop all nan columns


### ONLY CORTEX
cup_cortex = cup_result[cup_result['level_0'].isin(cortex_ids)]


cor_df_cup = corr_analysis(cup_cortex, target_col='density_W')


#%% Also concatenate gene name
cor_df_cup.rename(columns={'gene': 'entrez'}, inplace=True)
cor_df_cup = cor_df_cup.merge(conversion, on='entrez', how='left')  ### left keeps all values from result df
cor_df_cup = cor_df_cup.sort_values(by='correlation', ascending=False)

# import pickle as pkl
# with open(sav_fold + 'correlation_ISH_OL_density', 'wb') as file:
#     pkl.dump(df_merged, file)


#%% Get 3D gene map
goi = 'Zfpm2'
gene_map_3D = get_3D_gene_map(cup_result, cor_df_cup, goi, scale=10000)

plt.figure(); plt.imshow(gene_map_3D[300])




zzz

import napari
viewer = napari.Viewer()
viewer.add_image(gene_map_3D)
viewer.show(block=True)





""" SEE WHICH GENES SHIFT IN CORRELATION COMPARED BETWEEN CONTROL AND CUP PATTERNS? """





#%% Recovery as well

exp_name = 'cup ISH'


cup_df = df_concat[df_concat['exp'].isin(['P60', 'Cuprizone', 'Recovery', 'Cuprizone_NEW', 'Recovery_NEW'])]
df_means = cup_df.groupby(df_means.index).mean(numeric_only=True)
df_means['acronym'] = keys_df['acronym']
df_means['dataset'] = keys_df['dataset']
df_means['names'] = keys_df['names']
df_means['children'] = keys_df['children']


