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


""" GUI for loading a single file for Huganir lab synapse tracking """
def single_file_GUI():
    sg.theme('DarkAmber')   # Add a touch of color
    # All the stuff inside your window.
    layout = [[sg.Button('Select input file'), sg.Button('Cancel')]]
    
    # Create the Window
    window = sg.Window('Select input file', layout)
    
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            print('Exiting analysis')
            window.close()
            exit()
            break
    
        if event == 'Select input file':
            window.close()
    
            """ Select multiple folders for analysis AND creates new subfolder for results output """
            another_folder = True
            filenames = []
            while another_folder:
                root = tkinter.Tk()
                # get input folders
                
                input_path = "./"; initial_dir = './'
            
                # input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
                #                                     title='Please select input directory')
                # input_path = input_path + '/'
                
                # print('\nSelected directory: ' + input_path)
    
                filename = filedialog.askopenfilename(parent=root, initialdir= initial_dir,
                                                     title='Please select input file')
                print('\nSelected file: ' + filename)
    
                # 2nd layout
                layout2 = [[sg.Button('Select another file'), sg.Button('Select save directory')]]
                window2 = sg.Window('Select input file', layout2)
                event, values = window2.read()            
                if event == sg.WIN_CLOSED:
                    another_folder = False
                    print('Exiting analysis')
                    exit()
                    break             
                elif event == 'Select save directory': # if user closes window or clicks cancel
                    another_folder = False
                    
                elif event == 'Select another file':
                    another_folder = True
                
                window2.close()
    
                    
                filenames.append(filename)
                initial_dir = input_path
                
            break
       
    save_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
                                        title='Please select save directory')
    save_path = save_path + '/'          
 

    return filenames, save_path








def XTC_track_GUI(default_XY='0.83', default_Z='3'):
    sg.theme('DarkAmber')   # Add a touch of color
    # All the stuff inside your window.
    layout = [#[sg.Text('SNR warning thresh (optional)'), sg.InputText()],
              [sg.Text('XY resolution (um/pixel)'), sg.InputText(default_XY)],
              [sg.Text('Z resolution (um/pixel)'), sg.InputText(default_Z)],
              [sg.Button('Select input folder'), sg.Button('Cancel')]]
    
    # Create the Window
    window = sg.Window('XTC and tracking input', layout)
    
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
                window2 = sg.Window('XTC and tracking input', layout2)
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

