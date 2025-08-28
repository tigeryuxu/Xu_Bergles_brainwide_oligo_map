"""
    Generate GUI for SegCNN and TrackCNN
"""

import PySimpleGUI as sg
from sys import exit
import tkinter
from tkinter import filedialog

import glob, os
""" check if on windows or linux """
if os.name == 'posix':  platform = 'linux'
elif os.name == 'nt': platform = 'windows'
else:
    platform = 0



def seg_CNN_GUI(default_XY='0.83', default_Z='3'):
    sg.theme('DarkAmber')   # Add a touch of color
    # All the stuff inside your window.
    layout = [#[sg.Text('SNR warning thresh (optional)'), sg.InputText()],
              [sg.Text('XY resolution (um/pixel)'), sg.InputText(default_XY)],
              [sg.Text('Z resolution (um/pixel)'), sg.InputText(default_Z)],
              [sg.Button('Select input folder'), sg.Button('Cancel')]]
    
    # Create the Window
    window = sg.Window('Seg_CNN input', layout)
    
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        
        ### save user input values
        XY_res = values[0]
        Z_res = values[1]
        
        
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            print('Exiting analysis')
            window.close()
            exit()
            break
    
        if event == 'Select input folder':
            window.close()
    
            """ Select multiple folders for analysis AND creates new subfolder for results output """
            another_folder = True
            list_folder = []
            initial_dir = './'
            while another_folder:
                root = tkinter.Tk()
                # get input folders
                
                input_path = "./"; 
            
                input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
                                                    title='Please select input directory')
                input_path = input_path + '/'
                
                initial_dir = input_path
                
                print('\nSelected directory: ' + input_path)
    
    
                # 2nd layout
                layout2 = [[sg.Button('Select another folder'), sg.Button('Start analysis')]]
                window2 = sg.Window('Seg_CNN input', layout2)
                event, values = window2.read()            
                if event == sg.WIN_CLOSED:
                    another_folder = False
                    print('Exiting analysis')
                    exit()
                    break             
                elif event == 'Start analysis': # if user closes window or clicks cancel
                    another_folder = False
                    
                elif event == 'Select another folder':
                    another_folder = True
                
                window2.close()
    
                    
                list_folder.append(input_path)
                initial_dir = input_path
                
            break

    return list_folder, XY_res, Z_res


""" GUI for track-CNN """
def track_CNN_GUI(default_z='120', default_edge='40', default_obj_size='100'):
    sg.theme('DarkAmber')   # Add a touch of color
    # All the stuff inside your window.
    layout = [[sg.Text('Lowest z slice to analyze'), sg.InputText(default_z)],
              [sg.Text('Exclude edges of image (px)'), sg.InputText(default_edge)],
              [sg.Text('Minimum object size (px)'), sg.InputText(default_obj_size)],
              [sg.Button('Select input folder'), sg.Button('Cancel')]]
    
    # Create the Window
    window = sg.Window('Track_CNN input', layout)
    
    # Event Loop to process "events" and get the "values" of the inputs
    lowest_z_depth = []
    exclude_side_px = []
    min_size = []
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            print('Exiting analysis')
            window.close()
            exit()
            break
        
        else:
            lowest_z_depth = int(values[0])
            exclude_side_px = int(values[1])
            min_size = int(values[2])
            
    
        if event == 'Select input folder':
            window.close()
    
            """ Select multiple folders for analysis AND creates new subfolder for results output """
            another_folder = True
            list_folder = []
            while another_folder:
                root = tkinter.Tk()
                # get input folders
                
                input_path = "./"; initial_dir = './'
            
                input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
                                                    title='Please select input directory')
                input_path = input_path + '/'
                
                print('\nSelected directory: ' + input_path)
    
    
                # 2nd layout
                layout2 = [[sg.Button('Select another folder'), sg.Button('Start analysis')]]
                window2 = sg.Window('Seg_CNN input', layout2)
                event, values = window2.read()            
                if event == sg.WIN_CLOSED:
                    another_folder = False
                    print('Exiting analysis')
                    exit()
                    break             
                elif event == 'Start analysis': # if user closes window or clicks cancel
                    another_folder = False
                    
                elif event == 'Select another folder':
                    another_folder = True
                
                window2.close()
    
                    
                list_folder.append(input_path)
                initial_dir = input_path
                
            break

    return list_folder, lowest_z_depth, exclude_side_px, min_size



