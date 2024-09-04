#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:40:39 2024

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




def compare_GLOBAL_regions(df_global, names_to_plot, palette, xlim, ylim, ylim_norm, sav_name, sav_fold, exp_name, fontsize=14, add_zero=False, figsize=(4,3)):

    styles = ['-', '--', '-.', ':']
    

    plt.figure(figsize=figsize)
    ax1 = plt.gca()
    plt.figure(figsize=figsize)
    ax2 = plt.gca()
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = df_global[df_global['acronym'].str.fullmatch(name) == True]   
        
        
        ### DROP LAST TIMEPOINT FOR UNCERTAIN AREAS           
        if name == 'STR' or name == 'IB' or name == 'fiber tracts':
            match = match[match['exp'] != 'P800']  
                

        match['density_NORM'] = match['density_W']/np.nanmean(match.iloc[np.where(match['exp'] == 'P60')[0]]['density_W'])

        sns.lineplot(ax=ax2, x=match['age'], y=match['density_NORM'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                     errorbar='se')

        if add_zero:
            zero_df = {'exp':'P0', 'age':0, 'density_W':0, 'density_NORM':0, 'num_OLs_W':0, 'atlas_vol_W':1}
            zero_df = pd.DataFrame(zero_df, index=[5])            
            match = pd.concat([match, zero_df]).fillna(0)
            

        sns.lineplot(ax=ax1, x=match['age'], y=match['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                     errorbar='se')



    # ax = plt.gca()
    plt.sca(ax1)
    sns.move_legend(ax1, loc='upper left', frameon=False, title='', fontsize=fontsize)
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    
    plt.ylim([0, ylim])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ### set scientific notation
    ax1.ticklabel_format(axis='y', scilimits=(-4, 4))
    ax1.xaxis.get_offset_text().set_fontsize(fontsize-2)
    
    
    # Shrink current axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    plt.xlabel('Age (days)', fontsize=fontsize)
    
    plt.tight_layout()
    
    plt.savefig(sav_fold + exp_name + '_COMPARE_OVERALL_REGIONS_' + sav_name + '.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_COMPARE_OVERALL_REGIONS_' + sav_name + '.svg', format='svg', dpi=300)
    
    
    
    

    plt.sca(ax2)
    sns.move_legend(ax2, loc='upper left', frameon=False, title='', fontsize=fontsize)
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    
    plt.ylim(ylim_norm)
    plt.xlim(xlim)
        
    # Shrink current axis by 20%
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    # Put a legend to the right of the current axis
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    


    
    plt.ylabel('Normalized density', fontsize=fontsize)
    plt.xlabel('Age (days)', fontsize=fontsize)
    plt.tight_layout()
    # plt.margins(y=0)  ### prevent out of bounds (adding blank bars above and below graph)

    plt.savefig(sav_fold + exp_name + '_COMPARE_OVERALL_REGIONS_NORMALIZED_' + sav_name + '.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_COMPARE_OVERALL_REGIONS_NORMALIZED_' + sav_name + '.svg', format='svg', dpi=300)
    
    
from scipy import stats


def volcano_compare(df_compare, keys_df, compareby, group1, group2, fontsize=14, xplot='log2fold', thresh_pval=0.01, thresh_log2=0.3, xlim=2, ylim=6, figsize=(4,4)):

    all_dicts = []
    for id_k in keys_df['ids']:
                
        compare = df_compare.iloc[np.where(df_compare['ids'] == id_k)[0]]
        
        if len(compare) == 0:
            print('Skip, region not found')
            continue
        
        ### skip lower parcellations
        if compare.iloc[0]['st_level'] > 9:
            continue
        
        region = compare.iloc[0]['acronym']

        if np.isnan(compare['density_W']).all():
            continue
        
        x1vals = compare.iloc[np.where(compare[compareby] == group1)[0]]['density_W']
        x1 = np.mean(x1vals)
        
        x2vals = compare.iloc[np.where(compare[compareby] == group2)[0]]['density_W']
        x2 = np.mean(x2vals)
        
        tstat, pval = stats.ttest_ind(x1vals, x2vals, equal_var=True, alternative='two-sided')
        
        
        log2fold = np.mean(np.log2(x2)) - np.mean(np.log2(x1))
        
        change_type = 'none'
        if log2fold > thresh_log2 and pval < thresh_pval:
            change_type = 'up'
        elif log2fold < thresh_log2 * -1 and pval < thresh_pval:
            change_type = 'down'
            
        
        dict_pval = {'acronym': region, 'p_val': pval, 'mean_x1': x1, 'mean_x2': x2, 
                     'fold_change': x2/x1, 'abs_change': (x2 - x1), 'log2fold':log2fold, 'change_type':change_type}
        
        all_dicts.append(dict_pval)

    df_pval = pd.DataFrame(all_dicts)
        
    df_pval['-log10pval'] = np.log10(df_pval['p_val'])  * -1

    # thresh_pval = 0.01 ### pvalue of 0.01
    # thresh_fold = 0.2
    # thresh_abs = 5000
    
    # thresh_log2 = 0.5   ### where log2 (foldchange == 2) == 1, so value of 1 means A is 2 times B. Here 0.5 is somewhere around A is 1.5 times B
    palette = ['tab:gray', 'tab:green', 'tab:red']
    
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(df_pval, x=xplot, y='-log10pval', hue='change_type', palette=palette,
                         s=30)
    
    
    ax.axhline(np.log10(thresh_pval) * -1, linewidth=1, color='k', ls='--')
    ax.axvline(thresh_log2, linewidth=1, color='k', ls='--')
    ax.axvline(thresh_log2 * -1, linewidth=1, color='k', ls='--')
        
    
    outlier_ids = np.where(((df_pval[xplot] > thresh_log2) | (df_pval[xplot] < -thresh_log2)) & (df_pval["p_val"] < thresh_pval)  )[0]
    outliers = df_pval.iloc[outlier_ids]
    
    all_texts = []
    for idx, row in outliers.iterrows():
        all_texts.append(plt.text(row[xplot] + 0.01, row['-log10pval'] + 0.01, 
                  row['acronym'], ha='center', va='center',
                          size=10, color='black', weight='normal'))


    sns.move_legend(ax, loc='upper left', frameon=False, title='', fontsize=8)
    plt.xlim([-xlim, xlim])
    plt.ylim([0, ylim])
    plt.yticks(fontsize=fontsize - 2)
    # plt.yticks([])
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel('-Log10 (Pvalue)', fontsize=fontsize)
    plt.xlabel('Log2 (Fold Change)', fontsize=fontsize)
    plt.tight_layout()

    adjust_text(all_texts, #objects=ax, #only_move='x+',
                arrowprops=dict(arrowstyle='->', color='k'))
    
    
    
    
    
    
    


### First start with simple side-by-side plots...
def plot_stripes_from_df(df_means, df_concat, exp_name, reg_name, dname, to_remove, lvl_low, lvl_high, palette, figsize, leg_loc, sav_fold, fontsize, x_lab, y_lab, xlim, name, to_remove_substring='', logscale=False, dropna=False):
    ### get things to plot
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, df_concat, reg_name, dname, to_remove, to_remove_substring, lvl_low, lvl_high)
    
    
    if dropna:
        plot_vals = plot_vals[plot_vals[dname].notna()]
    
    plt.figure(figsize=figsize)

    sns.boxplot(x=plot_vals[dname], y=plot_vals['acronym'], hue=plot_vals['exp'], order=names_to_plot, palette=palette, 
                linecolor='gray', linewidth=1, fliersize=0)
    sns.stripplot(x=plot_vals[dname], y=plot_vals['acronym'], hue=plot_vals['exp'], order=names_to_plot, palette=palette, 
                  legend=False, dodge=True, size=6) #alpha=1, linewidth=1, ec='grey', size=5)   ### prevent plotting all dots in a straightline
                    

    for y in range(0, len(plot_vals['acronym'].unique())): 
        plt.axhspan(y - 0.5, y + 0.5, facecolor='black', alpha=[0.08 if y%2 == 0 else 0][0])
    
    ax = plt.gca()
    sns.move_legend(ax, loc=leg_loc, frameon=False, title='', fontsize=fontsize)
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if logscale:
        ax.set_xscale('log')

    else:
        ### set scientific notation
        ax.ticklabel_format(axis='x', scilimits=(-4, 4))
        ax.xaxis.get_offset_text().set_fontsize(fontsize-2)

    plt.xlim(xlim)
    plt.ylabel('')
    plt.xlabel(x_lab, fontsize=fontsize)
    plt.tight_layout()
    plt.margins(y=0)  ### prevent out of bounds (adding blank bars above and below graph)
    
    plt.savefig(sav_fold + exp_name +'_' + dname + '_' +  reg_name + '_' + name + '.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name +'_' + dname + '_' +  reg_name + '_' + name + '.svg', format='svg', dpi=300)
    
    return plot_vals




def plot_global_LR(pooled_df, names_to_plot, palette, sav_fold, exp_name, ylim, figsize, fontsize=14):
    all_melted = []    
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = pooled_df[pooled_df['acronym'].str.fullmatch(name) == True]

        
        match = match[match['exp'].str.fullmatch('P60') == True]
        
        melted = pd.melt(match, id_vars=['acronym'], value_vars=['density_L', 'density_R'])
        
        all_melted.append(melted)
        
        
    df_melt = pd.concat(all_melted)
    plt.figure(figsize=figsize)
    sns.boxplot(x=df_melt['acronym'], y=df_melt['value'], hue=df_melt['variable'], palette=palette, 
                #linecolor='gray', 
                fill=False,
                linewidth=1, fliersize=0)
    sns.stripplot(x=df_melt['acronym'], y=df_melt['value'], hue=df_melt['variable'], order=names_to_plot, palette=palette, 
                  legend=False, dodge=True, size=4) #alpha=1, linewidth=1, ec='grey', size=5)   ### prevent plotting all dots in a straightline

    
    plt.xticks(rotation=90)

    for x in range(0, len(df_melt['acronym'].unique())): 
        plt.axvspan(x - 0.5, x + 0.5, facecolor='black', alpha=[0.08 if x%2 == 0 else 0][0])
        
    ax = plt.gca()
    sns.move_legend(ax, loc='lower right', frameon=False, title='', fontsize=fontsize)
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    ### set scientific notation
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    ax.xaxis.get_offset_text().set_fontsize(fontsize-2)


    plt.yticks(np.arange(0, ylim+1, 5000))
    plt.ylim([0, ylim])        
    plt.xlabel('')
    # plt.xlabel(x_lab, fontsize=fontsize)
    plt.tight_layout()
    # plt.margins(y=0)  ### prevent out of bounds (adding blank bars above and below graph)

    plt.savefig(sav_fold + exp_name + '.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '.svg', format='svg', dpi=300)
    
 


### Plot horizontal organized plots
def plot_global(plot_df, names_to_plot, palette, sav_fold, exp_name, ylim, figsize, dname='density_W', fontsize=14, norm_name='', dropna=False, logscale=False):
    all_melted = []    
    for i_n, name in enumerate(names_to_plot):
        # if name matches FULLY, then go and plot
        match = plot_df[plot_df['acronym'].str.fullmatch(name) == True]

        if len(norm_name) > 0:
            norm = match[match['exp'].str.contains(norm_name) == True]
            norm = np.mean(norm[dname])
            
            match[dname] = match[dname]/norm


        all_melted.append(match)
        
        
        
  
    df_melt = pd.concat(all_melted)
    
    if dropna:
        df_melt = df_melt[df_melt[dname].notna()]
    
    plt.figure(figsize=figsize)
    sns.boxplot(x=df_melt['acronym'], y=df_melt[dname], hue=df_melt['exp'], palette=palette, 
                #linecolor='gray', 
                fill=False,
                linewidth=1, fliersize=0)
    sns.stripplot(x=df_melt['acronym'], y=df_melt[dname], hue=df_melt['exp'], order=names_to_plot, palette=palette, 
                  legend=False, dodge=True, size=4) #alpha=1, linewidth=1, ec='grey', size=5)   ### prevent plotting all dots in a straightline

    
    plt.xticks(rotation=90)

    for x in range(0, len(df_melt['acronym'].unique())): 
        plt.axvspan(x - 0.5, x + 0.5, facecolor='black', alpha=[0.08 if x%2 == 0 else 0][0])
        
    ax = plt.gca()
    sns.move_legend(ax, loc='upper left', frameon=False, title='', fontsize=fontsize)
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    
    if logscale:
        ax.set_yscale('log')
        print('logscale')
        plt.ylim([0, 1000000])
    else:
        plt.ylim([0, ylim])
        ### set scientific notation
        ax.ticklabel_format(axis='y', scilimits=(-4, 4))
        ax.xaxis.get_offset_text().set_fontsize(fontsize-2)




    
    plt.xlabel('')
    # plt.xlabel(x_lab, fontsize=fontsize)
    plt.tight_layout()
    # plt.margins(y=0)  ### prevent out of bounds (adding blank bars above and below graph)

    plt.savefig(sav_fold + exp_name + '.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '.svg', format='svg', dpi=300)
        
    
    
### Plots each layer individually (NOT OVER TIME)    
def plot_by_layer(plot_df, compare_df, names_to_plot, sav_fold, exp_name, plot_name, dname='density_W', figsize=(4.8,4), ylim=[0, 10000], fontsize=14):    
    plt.figure(figsize=figsize)

    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = plot_df[plot_df['acronym'].str.contains(name) == True]
        if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
            match = match[match['acronym'].str.contains('RSPd4') == False]   
        
        ### Compare and normalize to this baseline
        if len(compare_df) > 0:
            match_comp = compare_df[compare_df['acronym'].str.contains(name) == True]
            if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
                match_comp = match_comp[match_comp['acronym'].str.contains('RSPd4') == False]   
            
        
        layers = ['1', '2/3', '4', '5', '6']
        
        all_layers = []
        for layer in layers:
        
            lay_df = match[match['names'].str.contains(layer) == True]
        
            if len(lay_df) == 0:
                continue
        
            ### FOR CUPRIZONE SUBTRACT TO FIND DIFFERENCE (delta density i.e. density change)
        
            sum_df = lay_df.groupby(['dataset']).sum()
        
            # sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
            # sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
            if dname == 'density_LARGE_W_CLEAN':                
                sum_df['density_W'] = sum_df['num_large_W_CLEAN']/sum_df['atlas_vol_W']
                
                    
            
            
            ### Compare and normalize to this baseline
            if len(compare_df) > 0:
                
                # sum_df = sum_df.groupby(['ids']).sum()
                
                lay_df = match_comp[match_comp['names'].str.contains(layer) == True]
            
                sum_df_comp = lay_df.groupby(['dataset']).sum()
            
                # sum_df_comp['density_L'] = sum_df_comp['num_OLs_L']/sum_df_comp['atlas_vol_L']
                # sum_df_comp['density_R'] = sum_df_comp['num_OLs_R']/sum_df_comp['atlas_vol_R']
                sum_df_comp['density_W'] = sum_df_comp['num_OLs_W']/sum_df_comp['atlas_vol_W']
                
                
                if dname == 'density_LARGE_W':                
                    sum_df_comp['density_W'] = sum_df_comp['num_large_W_CLEAN']/sum_df_comp['atlas_vol_W']
                    
                    
                    
                
                baseline = np.mean(sum_df_comp['density_W'])
                
            
                ### normalize to baseline
                sum_df['density_W'] = sum_df['density_W']/baseline
            
        
            sum_df['layer'] = layer

            all_layers.append(sum_df)
            
        df_layers = pd.concat(all_layers)
            
        ### SKIP if wasnt subdivided into smaller layer units
        if df_layers['density_W'].isna().any():
            continue
            
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
    
    
    # ax.set_yscale('log')
    # plt.ylim([0, 100000])
    # plt.tight_layout()
    
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG.svg', format='svg', dpi=300)


    # ### Then plot it linear
    plt.legend(loc = 'upper left', frameon=False)
    ax.set_yscale('linear')
    #plt.ylim([0, 20000])  ### this ruins log plot...

    
    plt.ylim(ylim) 
    # plt.yticks(np.arange(0, 11000, step=5000))
    
    ax = plt.gca() 
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))    
    

    
    plt.tight_layout()
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_' + plot_name + '.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_' + plot_name + '.svg', format='svg', dpi=300)





# Start with ISOCORTEX - medium level
#to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|'
# to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|'
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Isocortex', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=9, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 25000], name='medium_level',
            
#             # to_remove_substring='6'
#             ) 
# print('Number of cortical cells: ' + str(plot_vals['num_OLs_W'].sum()))


# #%%% Compare layers for each cortical region
# all_layers = []
# for layer in layers:
#     plt.figure(figsize=fig_long)
#     for i_n, name in enumerate(names_to_plot):
#         # if name matches the first half at minimum, then go and plot
#         match = aging_df[aging_df['acronym'].str.contains(name) == True]
#         if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
#             match = match[match['acronym'].str.contains('RSPd4') == False]   
#         match=match.reset_index()
        

#         ### DROP layer 1 from M127 and M126 for the moment due to delipidation artifiact
#         # if layer == '1':
#         #     match = match.drop(index=np.where(match['dataset'] == 'M127')[0]).reset_index()        
#         #     match = match.drop(index=np.where(match['dataset'] == 'M126')[0])

#         lay_df = match[match['names'].str.contains(layer) == True]
 
        
#         sum_df = lay_df.groupby(['dataset', 'exp', 'age']).sum(numeric_only=True)
#         sum_df = sum_df.reset_index() ### moves all of the indices into columns!!!

#         sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']

#         sns.lineplot(x=sum_df['age'], y=sum_df['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)]   )

#         plt.title('Layer ' + layer)

#         # give name of layer
#         sum_df['layer'] = layer
    
#         all_layers.append(sum_df)
        
# df_layers = pd.concat(all_layers)

# df_layers = df_layers.groupby(['dataset', 'layer', 'age']).sum(numeric_only=True)     
# df_layers['density_W'] = df_layers['density_W']/df_layers['atlas_vol_W']   ### scale it down since it's sum above   

# df_layers = df_layers.reset_index()



#%%% Start with ISOCORTEX - medium level
#to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|'
# to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|'
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Isocortex', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=9, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 25000], name='medium_level',
            
#             # to_remove_substring='6'
#             ) 
# print('Number of cortical cells: ' + str(plot_vals['num_OLs_W'].sum()))
    

# ### PLOT at higher level
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Isocortex', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=(6,9), lvl_low=5, lvl_high=11, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 30000], name='lower_level') 


# ### PLOT at lower level
# to_remove = 'MO|SS|AUDp|AUDd|VISpl|VISrl|AUDv|VISal|VISp|VISl|VISa|AUDpo|VISpor|VISam|VISli|VISpm'

# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove, reg_name = 'Isocortex', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=9, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 25000], name='high_level') 


# ### Absolute # of OLs
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Isocortex', dname='num_OLs_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=9, leg_loc='lower right',
#             x_lab='Number of OLs', y_lab='', xlim=[0, 1000000], name='_NUM_OLS_ABS', logscale=True) 




# #%%% Get coordinates of cells in hippocampus to plot

# to_remove_HIPPO = 'HPF|HIP|RHP|CA|ProSv|ProS'
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_HIPPO, reg_name = 'Hippocampal formation', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=6, lvl_high=9, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 40000], name='medium_level') 
# print('Number of hippocampal cells: ' + str(plot_vals['num_OLs_W'].sum()))





# #%%% Get coordinates of cells in thalamus to plot
# to_remove_THAL = 'Th|'
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_THAL, reg_name = 'Thalamus', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=(4.5, 3.5), lvl_low=5, lvl_high=8, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 100000], name='medium_level') 
# print('Number of Thalamic cells: ' + str(plot_vals['num_OLs_W'].sum()))


# to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|VISpl|'
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Thalamus, sensory-motor cortex related', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=11, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 80000], name='SENSORY_THALAMUS') 
# print('Number of cortical cells: ' + str(plot_vals['num_OLs_W'].sum()))



# #%%% Get coordinates of cells in striatum to plot
# to_remove_STRI = ''
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_STRI, reg_name = 'Striatum', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=(6,4), lvl_low=5, lvl_high=8, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 35000], name='medium_level') 
# print('Number of striatal cells: ' + str(plot_vals['num_OLs_W'].sum()))

    

# #%%% Get coordinates of cells in Midbrain to plot
# to_remove_MIDBRAIN = ''
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_MIDBRAIN, reg_name = 'Midbrain', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=8, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 80000], name='medium_level', logscale=False) 
# print('Number of midbrain cells: ' + str(plot_vals['num_OLs_W'].sum()))



# #%%% Get coordinates of cells in Interbrain to plot
# to_remove_INTERBRAIN = ''
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_INTERBRAIN, reg_name = 'Hypothalamus', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=8, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 60000], name='medium_level', logscale=False) 
# print('Number of interbrain cells: ' + str(plot_vals['num_OLs_W'].sum()))
    

# #%%% Get coordinates of cells in fiber tracts to plot

# to_remove_TRACTS = 'cbp|drt'
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_TRACTS, reg_name = 'fiber tracts', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=8, leg_loc='lower right',
#             x_lab='Density (cells/mm\u00b3)', y_lab='', xlim=[0, 100000], name='medium_level', logscale=True) 
# print('Number of fiber tracts cells: ' + str(plot_vals['num_OLs_W'].sum()))

    




# #%%% Get coordinates of cells in Cerebellum to plot
# to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|VISpl|'
# plot_vals = plot_stripes_from_df(df_means, aging_df, exp_name, to_remove=to_remove_CORTEX, reg_name = 'Cerebellar cortex', dname='density_W',
#             sav_fold=sav_fold, fontsize=fontsize, palette=palette, figsize=fig_stand, lvl_low=5, lvl_high=8, leg_loc='lower right',
#             x_lab='Density of OLs (cells/mm\u00b3)', y_lab='', xlim=[0, 30000], name='medium_level') 




#%%% Get coordinates of cells in hypothalamus to plot

# boxplot_by_subkey(df_means, aging_df, reg_name='Hypothalamus', dname='density_W', to_remove='Th|', lvl_low=5, lvl_high=8)
# # boxplot_by_subkey(df_means, aging_df, reg_name='Isocortex', dname='num_OLs_W', to_remove='MO|SS|VIS|GU|AUD', lvl_low=5, lvl_high=9)

# plot_vals, names_to_plot = get_subkeys_to_plot(df_means, aging_df, reg_name='Thalamus', dname='density_W', to_remove='Th|', lvl_low=5, lvl_high=8)

# palette = sns.color_palette("Set2")
# plt.figure()
# sns.stripplot(x=plot_vals['density_W'], y=plot_vals['acronym'], hue=plot_vals['exp'], order=names_to_plot, palette=palette, legend=False)
# sns.boxplot(x=plot_vals['density_W'], y=plot_vals['acronym'], hue=plot_vals['exp'], order=names_to_plot, palette=palette)
# ax = plt.gca()
# sns.move_legend(ax, loc='lower right', frameon=False, title='')
# plt.yticks(fontsize=fontsize)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# #plt.xlim([0, 43000])
# plt.ylabel('')
# plt.xlabel('Density (cells/mm\u00b3)', fontsize=fontsize + 2)
# plt.tight_layout()
# plt.savefig(sav_fold + exp_name + '_THALAMUS_medium_level.png', dpi=300)



### GET OTHER REGIONS - orbital, gustatory, subcortical, white matter tracts...



#%%% Extract ALL layer 4? And unfurl it... solve LaPlace equation and project it?

#https://www.cell.com/cell-reports/fulltext/S2211-1247(22)00764-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2211124722007641%3Fshowall%3Dtrue

#https://pubmed.ncbi.nlm.nih.gov/35732133/