#!/usrbin/env python3
# -*- coding: utf-8 -*-
"""
Syglass to MaMut Code Converter

@author: Ephraim Musheyev for the Bergles Lab (2022)
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

"""
INSTRUCTIONS:
Converts syGlass CSV files to MaMut annotation XML file and assigns different colors for newly formed and/or dying/dividing cells

1. Copy the pathname of your syGlass CSV file into the "syglass_path" variable
2. Enter the MaMut XML file name into the "xml_file_name" variable 
3. Enter the pathname of the folder where you want to save the annotation XML file (the annotated file eventually needs to be in the same folder as the H5-XML file pair) into the "destination_path" variable
4. Enter the patname of the "MaMut_Shell_File.xml" file located in the main folder

*To see the different colors of cells, make sure the "Color Spots By" dropdown menu in MaMut is set to "Manual Spot Color"*
"""

"""
USER INPUT:
"""

syglass_path = '/media/user/FantomHD/710_invivo_imaging/MoE_in_vivo_blood_vessel/20220615_MoE_blood_vessel_CLEMASTINE/SimpleElastix_reg/converted_mamut_files/A_MW43_FOV1_tracked_cells_df_POST_PROCESSED_SYGLASS.csv' # <- Enter syGlass CSV pathname here (as a string)
xml_file_name = 'A_MW43_FOV1_BDV_export' # <- Enter the H5 Filename here (as a string, just the filename, not the .h5 extension)
destination_path = '/media/user/FantomHD/710_invivo_imaging/MoE_in_vivo_blood_vessel/20220615_MoE_blood_vessel_CLEMASTINE/SimpleElastix_reg/converted_mamut_files/' # <- Enter destination pathname here (as a string)
shell_path = '/home/user/Documents/GitHub/Xu_Bergles_2021_Oligo_Track/Oligo_Track/SyGlass_MaMut_Converter/MaMut_Shell_File.xml' # <- Enter the "MaMut_Shell_File.xml" pathname here (as a string)


# xy_scale = 0.5930477
# z_scale = 2


syglass_path = '/media/user/FantomHD/710_invivo_imaging/MoE_in_vivo_TIE2CRE_blood_vessels/MOe TIE2 CRE vascular WT/OligoTrack/A_MW84_FOV1/A_MW84_FOV1_output_seg_CNN/A_MW84_FOV1_output_seg_CNN_output_Track_CNN/tracked_cells_df_POST_PROCESSED_SYGLASS.csv' # <- Enter syGlass CSV pathname here (as a string)
xml_file_name = 'MW84_FOV1_BDV_export' # <- Enter the H5 Filename here (as a string, just the filename, not the .h5 extension)
destination_path = '/media/user/FantomHD/710_invivo_imaging/MoE_in_vivo_TIE2CRE_blood_vessels/MOe TIE2 CRE vascular WT/OligoTrack/A_MW84_FOV1/MaMut/' # <- Enter destination pathname here (as a string)


xy_scale = 0.4994082
z_scale = 2

"""
CONVERTER:
"""

# Reads in the syGlass CSV
df= pd.read_csv(syglass_path)
num_rows = df.shape[0] #gets passed to the AllSpots nspots element


zzz



"""
Rewrites the 'COLOR' column of the syglass file to log
which cells appear after the first frame and which cells 
dissappear before the final fram (which cells are born and which cells die/divide)
"""
def spot_color():

    # Used to extract the series numbers that actually have data in the CSV
    min_series = df['SERIES'].min()
    max_series = df['SERIES'].max()
    
    # Used to compare the frames in each series to the nubmer of frames in the entire image
    min_frame = df['FRAME'].min()
    max_frame = df['FRAME'].max()
    
    # A list that contains all the series numbers that correspond to data
    series_vals_list = [l for l in range(min_series,max_series+1) if df[df['SERIES'] == l].shape[0] >= 1]
 
    new_cell_list = []
    old_cell_list = []
    both_cell_list = []
 
    # Loops through all the series numbers and figures out which cells aren't present for the entire timeframe
    for p in series_vals_list:

        # Makes a new dataframe that isolates each series
        df_series = df[df['SERIES'] == p] 
        index_values = df_series.index.values

        # Determines if the cell appears after the first frame and is present at the end and logs it's series in the new_cell_list
        if df_series['FRAME'].min() > min_frame and df_series['FRAME'].max() == max_frame:
            new_cell_list.extend(index_values)
            
        # Determines if the cell is present at the start and disappears before the last frame and logs it's series in the old_cell_list
        if df_series['FRAME'].max() < max_frame and df_series['FRAME'].min() == min_frame:
            old_cell_list.extend(index_values)
            
        if df_series['FRAME'].max() < max_frame and df_series['FRAME'].min() > min_frame:
            both_cell_list.extend(index_values)
    
    # Loops through the new and old cell lists and changes the color in the dataframe (0=new, 1=old)
    for n in range(0, len(new_cell_list)):
        df.loc[new_cell_list[n], 'COLOR'] = 0
    for m in range(0, len(old_cell_list)):
        df.loc[old_cell_list[m], 'COLOR'] = 1
    for b in range(0, len(both_cell_list)):
        df.loc[both_cell_list[b], 'COLOR'] = 2

    # Returns updated dataframe with logged color information
    return df

"""
Converts all the individual datapoints in the CSV to
MaMut-style XML format

Datapoints are arranged according to their timepoint
(All datapoints at t=0 are grouped under one SpotsInFrame element)
"""
def spots_conversion(xy_scale, z_scale):
    
    # Creating AllSpots element, which contains all SpotsInFrame elements
    AllSpots_element = ET.Element('AllSpots')
    AllSpots_element.set('nspots',str(num_rows))

    min_frame = df['FRAME'].min()
    max_frame = df['FRAME'].max()
    
    # Creates SpotsInFrame element for each timepoint
    for i in range(min_frame,max_frame+1):

        SpotsInFrame_element = ET.Element('SpotsInFrame')
        SpotsInFrame_element.set('frame',str(i))
        df_i = df[df['FRAME'] == i]

        # Creates Spots subelement, which contains all the datapoints at each SpotsInFrame timeframe
        for index, row in df_i.iterrows():
            
            # Extracts relevant data from CSV for each datapoint
            x=row['X'] * xy_scale
            y=row['Y'] * xy_scale
            z=row['Z'] * z_scale
            frame=row['FRAME']
            series=row['SERIES']  
            color=row['COLOR']

            """
            A Spot ID is assigned to each spot
            It is formed by adding a 0 the beginning of the frame number and series number, and combining the two numbers 
            (ex. frame 3 of series 34 would become "03034")
            """
            id_=(f'0{frame}0{series}')
            
            # Library for assigning colors to the spots later on
            blue='-13421569' # New cell
            red='-52429' # Old cell
            purple = '-3407668' #
            green='-13369549' # All else
            
            # Creating Spots subelement and setting its atributes
            Spots_element=ET.SubElement(SpotsInFrame_element, 'Spot')
        
            Spots_element.set('ID', f'{id_}')
            Spots_element.set('name',f'ID{id_}')  
            Spots_element.set('STD_INTENSITY_CH1','0.0')  
            Spots_element.set('STD_INTENSITY_CH2','0.0')  
            Spots_element.set('QUALITY','-1.0') 
            Spots_element.set('POSITION_T',f'{frame}') 
            Spots_element.set('TOTAL_INTENSITY_CH2','0.0') 
            Spots_element.set('TOTAL_INTENSITY_CH1','0.0') 
            Spots_element.set('CONTRAST_CH1','0.0') 
            Spots_element.set('FRAME',f'{frame}.0') 
            Spots_element.set('CONTRAST_CH2','0.0') 
            Spots_element.set('MEAN_INTENSITY_CH1','0.0') 
            Spots_element.set('MAX_INTENSITY_CH2','0.0') 
            Spots_element.set('MEAN_INTENSITY_CH2','0.0') 
            Spots_element.set('MAX_INTENSITY_CH1','0.0') 
            Spots_element.set('SOURCE_ID','0') 
            Spots_element.set('MIN_INTENSITY_CH2','0.0') 
            Spots_element.set('MIN_INTENSITY_CH1','0.0') 
            Spots_element.set('SNR_CH1','0.0') 
            Spots_element.set('SNR_CH2','0.0') 
            Spots_element.set('MEDIAN_INTENSITY_CH1','0.0') 
            Spots_element.set('VISIBILITY','1') 
            Spots_element.set('RADIUS','10.0')
            
            #Assigning color to the spots
            if color == 0:
                Spots_element.set('MANUAL_SPOT_COLOR',f'{blue}')
            elif color == 1:
                Spots_element.set('MANUAL_SPOT_COLOR',f'{red}')
            elif color == 2:
                Spots_element.set('MANUAL_SPOT_COLOR',f'{purple}')
            else:
                Spots_element.set('MANUAL_SPOT_COLOR',f'{green}')
    
            Spots_element.set('MEDIAN_INTENSITY_CH2','0.0') 
            Spots_element.set('POSITION_X',f'{x}') 
            Spots_element.set('POSITION_Y', f'{y}') 
            Spots_element.set('POSITION_Z',f'{z}') 
        
        AllSpots_element.append(SpotsInFrame_element)

    return AllSpots_element


"""
Creates tracks for linking datapoints of the same cell across the different time points
Tracks are organized by a Source ID number the connects to a Target ID number
These ID numbers are assigned in the same way as Spot ID nubmers
"""
def tracks_conversion():

    AllTracks_element = ET.Element('AllTracks')
    
    min_series = df['SERIES'].min()
    max_series = df['SERIES'].max()

    # Identifying all the series nubmers that have more than one datapoint that can be linked together
    series_vals_list = [l for l in range(min_series,max_series+1) if df[df['SERIES'] == l].shape[0] >= 2]

    # Used to assign the track numbers (increments by 1 for each subsequent track)
    j=0

    # A list of all the track numbers used in the filtered_tracks_conversion function
    Track_ID_list = []

    # For loop that makes a Track element for each eligble series
    for p in series_vals_list:
        
        # Makes a new dataframe that isolates each series
        df_series = df[df['SERIES'] == p] 

        # Resets the index of the dataframe so that the for loop can be properly stopped early
        df_series.reset_index(inplace=True, drop=True)
        num_rows = df_series.shape[0]

        # Creates the Track_element for each series
        Track_element = ET.Element('Track') 
        Track_element.set('name',f'Track_{j}') #does this need to be 1,2,3..., or can it be equal to p?
        Track_element.set('TRACK_ID',f'{j}')
        Track_element.set('TRACK_INDEX',f'{j}')
        Track_element.set('DIVISION_TIME','NaN')
        
        #makes a list of all the track IDs to use in the Filtered_Track function
        Track_ID_list.append(j)
        j=j+1

        # Populating the track_element with the information from each point in the series as an Edge subelement
        for index, row in df_series.iterrows(): 

            # Creating the source and target ID numbers
            frame=row['FRAME']
            series=row['SERIES']    
            next_row = df_series.loc[index+1]
            frame1=next_row['FRAME']
            series1=next_row['SERIES']
            source_id=(f'0{frame}0{series}')
            target_id=(f'0{frame1}0{series1}')
            
            # Creating and assigning attributed to the Edge subelement
            Edge_element=ET.SubElement(Track_element, 'Edge')
            
            Edge_element.set('SPOT_SOURCE_ID',f'{source_id}')
            Edge_element.set('SPOT_TARGET_ID',f'{target_id}')
            Edge_element.set('LINK_COST','-1.0')
            Edge_element.set('DIRECTIONAL_CHANGE_RATE','0.0')
            Edge_element.set('SPEED','0.0')
            Edge_element.set('DISPLACEMENT','0.0')
            Edge_element.set('EDGE_TIME','0.0')
            Edge_element.set('EDGE_X_LOCATION','0.0')
            Edge_element.set('EDGE_Y_LOCATION','0.0')
            Edge_element.set('EDGE_Z_LOCATION','0.0')
            
            #Break loop at second to last row (nothing to link the last datapoint to)
            if index == num_rows - 2:
                break

        AllTracks_element.append(Track_element)

    return AllTracks_element, Track_ID_list

"""
Creates the FilteredTracks component of the MaMut XML file that lists out all of the TrackID numbers
Uses the Track_ID_list variable created in the tracks_converted function
"""
def filtered_tracks_conversion(Track_ID_list):

    FilteredTracks_element = ET.Element('FilteredTracks')
    
    for k in range(0,len(Track_ID_list)):
        TrackID_element=ET.SubElement(FilteredTracks_element,'TrackID')
        TrackID_element.set('TRACK_ID',str(Track_ID_list[k]))

    return FilteredTracks_element

# Parse the MaMut shell xml file
tree = ET.parse(shell_path) 
root = tree.getroot()

# Locate the Model element of the XML file; Generating and appending the AllSpots, AllTracks, and Filtered Tracks elements
model_element = root.find('Model')
df = spot_color()
AllSpots= spots_conversion(xy_scale, z_scale)
AllTracks, Track_ID_list = tracks_conversion()
FilteredTracks = filtered_tracks_conversion(Track_ID_list)
model_element.append(AllSpots)
model_element.append(AllTracks)
model_element.append(FilteredTracks)

# Making sure that the outputs of each function are what they're supposed to be
xml_string1 = ET.tostring(AllSpots, encoding='unicode')
xml_string2 = ET.tostring(AllTracks, encoding='unicode')
xml_string3 = ET.tostring(FilteredTracks, encoding='unicode')

# Editing the name of the MaMut file name within the shell file
imagedata_element = root.find('.//ImageData')
imagedata_element.set('filename', f'{xml_file_name}.xml')

# Saving the XML file in the destinatino folder
tree.write(f'{destination_path}/{xml_file_name}_annotations.xml', encoding='utf-8', xml_declaration=True)

