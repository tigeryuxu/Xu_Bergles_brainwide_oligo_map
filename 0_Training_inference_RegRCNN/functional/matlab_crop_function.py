import numpy as np


""" EXTREMELY IMPORTANT CROPPING FUNCTION:
    
            given center point (x, y, z), will crop a FOV of size crop_size x crop_size x z_size around input_im
    """

def crop_around_centroid(input_im, y, x, z, crop_size, z_size, height, width, depth):
    
    
     """ ADDING + 1 shifts it to be 39, 39, 15 in Python coords, which matches with 40, 40, 16 in MATLAB coords

     
     """
    
    
     box_x_max = x + crop_size + 1; box_x_min = x - crop_size + 1;
     box_y_max = y + crop_size + 1; box_y_min = y - crop_size + 1;
     box_z_max = round(z + z_size/2) + 1; box_z_min = round(z - z_size/2) + 1;
     
     im_size_x = width;
     im_size_y = height;
     im_size_z = depth;
     
     
     ### Setup "x"
     
     if box_x_max > im_size_x:
         overshoot = box_x_max - im_size_x;
         box_x_max = box_x_max - overshoot;
         box_x_min = box_x_min - overshoot;
     
     if box_x_min < 0:
         overshoot_neg = (-1) * box_x_min + 1;
         box_x_min = box_x_min + overshoot_neg;
         box_x_max = box_x_max + overshoot_neg;
         
     elif box_x_min == 0:  ### Tiger added April 25th, 2021
         overshoot_neg = 0
         
     
     
     
     ### Setup "y"
     if box_y_max > im_size_y:
         overshoot = box_y_max - im_size_y;
         box_y_max = box_y_max - overshoot;
         box_y_min = box_y_min - overshoot;
     
     if box_y_min < 0:
         overshoot_neg = (-1) * box_y_min + 1;
         box_y_min = box_y_min + overshoot_neg;
         box_y_max = box_y_max + overshoot_neg;     

     elif box_y_min == 0:   ### Tiger added April 25th, 2021
         overshoot_neg = 0        

     
     ### Setup "z"
     
     if box_z_max > im_size_z:
         overshoot = box_z_max - im_size_z;
         box_z_max = box_z_max - overshoot;
         box_z_min = box_z_min - overshoot;
     
     if box_z_min < 0:
         overshoot_neg = (-1) * box_z_min + 1;
         box_z_min = box_z_min + overshoot_neg;
         box_z_max = box_z_max + overshoot_neg;

     elif box_z_min == 0:   ### Tiger added April 25th, 2021
         overshoot_neg = 0                


     box_x_max - box_x_min
     box_y_max - box_y_min
     box_z_max - box_z_min
     
     
     # box_x_min -= 1
     # box_x_max -= 1
     
     # box_y_min -= 1
     # box_y_max -= 1
     
     # box_z_min -= 1
     # box_z_max -= 1
     
     
     """ Python indexing requires minus 1 from all dimensions """
     crop = input_im[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max]
     where_are_NaNs = np.isnan(crop)
     crop[where_are_NaNs] = 0
     
     return crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max

     
     
     
     
def crop_around_centroid_with_pads(input_im, y, x, z, crop_size, z_size, height, width, depth):
     box_x_max = int(x + crop_size) + 1; box_x_min = int(x - crop_size) + 1;
     box_y_max = int(y + crop_size) + 1; box_y_min = int(y - crop_size) + 1;
     box_z_max = int(z + z_size/2) + 1; box_z_min = int(z - z_size/2) + 1;
     
     im_size_x = width;
     im_size_y = height;
     im_size_z = depth;
     
     
     overshoot_x = 0; overshoot_neg_x = 0;
     if box_x_max > im_size_x:
         overshoot_x = box_x_max - im_size_x;
         box_x_max = im_size_x;
         #box_x_min = box_x_min - overshoot;
     
     if box_x_min <= 0:
         overshoot_neg_x = (-1) * box_x_min;
         box_x_min = 0;
         #box_x_max = box_x_max + overshoot_neg;
     
     overshoot_y = 0; overshoot_neg_y = 0;
     if box_y_max > im_size_y:
         overshoot_y = box_y_max - im_size_y;
         box_y_max = im_size_y;
         #box_y_min = box_y_min - overshoot;
     
     if box_y_min <= 0:
         overshoot_neg_y = (-1) * box_y_min;
         box_y_min = 0;
         #box_y_max = box_y_max + overshoot_neg;     
     
     overshoot_z = 0; overshoot_neg_z = 0;
     if box_z_max > im_size_z:
         overshoot_z = box_z_max - im_size_z;
         box_z_max = im_size_z;
         #box_z_min = box_z_min - overshoot;
     
     if box_z_min <= 0:
         overshoot_neg_z = (-1) * box_z_min;
         box_z_min = 0;
         #box_z_max = box_z_max + overshoot_neg;
     
     box_x_max - box_x_min
     box_y_max - box_y_min
     box_z_max - box_z_min
     
     
     # box_x_min -= 1
     # box_x_max -= 1
     
     # box_y_min -= 1
     # box_y_max -= 1
     
     # box_z_min -= 1
     # box_z_max -= 1
     
     
     """ Python indexing requires minus 1 from all dimensions """
     crop = input_im[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max]
     where_are_NaNs = np.isnan(crop)
     crop[where_are_NaNs] = 0
     
     
     crop_pad = np.pad(crop, ((overshoot_neg_x, overshoot_x), (overshoot_neg_y, overshoot_y), (overshoot_neg_z, overshoot_z)))
     crop = crop_pad


     boundaries_crop = np.zeros(np.shape(crop_pad))
     boundaries_crop[overshoot_neg_x: crop_size * 2 - overshoot_x, overshoot_neg_y: crop_size * 2 - overshoot_y,  overshoot_neg_z: z_size - overshoot_z] = 1
     
     box_xyz = [box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max]
     
     box_over = [overshoot_neg_x, overshoot_x, overshoot_neg_y, overshoot_y, overshoot_neg_z, overshoot_z]
     
     return crop, box_xyz, box_over, boundaries_crop   
    
          
     
     
     
     
