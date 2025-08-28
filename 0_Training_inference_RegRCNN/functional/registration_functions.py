#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:36:16 2022

@author: user
"""


import sys,os
sys.path.append('/home/user/build/SimpleITK-build/Wrapping/Python/')
import SimpleITK as sitk
import numpy as np




""" For normal registration """
def register_ELASTIX(fixed_im, moving_im, reg_type='affine'):

    fixedImage = sitk.GetImageFromArray(fixed_im)
    movingImage = sitk.GetImageFromArray(moving_im)
    
    parameterMapVector = sitk.VectorOfParameterMap()
    
    ### for rigid
    parameterMapVector.append(sitk.GetDefaultParameterMap('translation'))
    
    
    ### for affine only
    if reg_type == 'affine':
        parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    
    elif reg_type == 'nonrigid':        
        ### for non-rigid
        parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
        
    else:
        print('required reg_type: "affine" or "nonrigid"')
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetLogToConsole(False)
    #elastixImageFilter.LogToConsoleOff()

    elastixImageFilter.Execute()
    
    resultImage = elastixImageFilter.GetResultImage()
    
    registered_im = sitk.GetArrayFromImage(resultImage)
    
    
    registered_im[registered_im < 0] = 0   ### set negative values to be 0
    registered_im[registered_im >= 255] = 255       
    
    
    ### also return transform so can do on new image
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transformParameterMap)
        
    
    return registered_im, transformix





""" For slice-by-slice registration """

def register_by_SLICE(fixed_im, moving_im, reg_type='affine', reapply_im=[]):
    reg_slices = []
    reapply_slices = []
    for slice_id in range(len(fixed_im)):
        
        
        slice_fixed = fixed_im[slice_id, :, :]
        slice_move = moving_im[slice_id, :, :]
        
        
        reg_im, transformix = register_ELASTIX(slice_fixed, slice_move, reg_type='affine')  ### can also be "nonrigid"

        reg_im = np.expand_dims(reg_im, 0)
        reg_slices.append(reg_im)
        
        if len(reapply_im) > 0:  ### if nothing to reapply to
            slice_reapply = reapply_im[slice_id, :, :]
            
            reapply_reg = reapply_transform_map(slice_reapply, transformix)
            reapply_reg = np.expand_dims(reapply_reg, 0)
            reapply_slices.append(reapply_reg)


    return reg_slices, reapply_slices, transformix
    

def reapply_transform_map(raw_moving, transformix):

    transformix.SetMovingImage(sitk.GetImageFromArray(raw_moving))
    transformix.Execute()
    raw_reg = sitk.GetArrayFromImage(transformix.GetResultImage())
    raw_reg[raw_reg < 0] = 0   ### set negative values to be 0
    raw_reg[raw_reg >= 255] = 255      


    return raw_reg
            