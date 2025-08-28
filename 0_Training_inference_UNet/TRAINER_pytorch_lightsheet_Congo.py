# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger


Add:
        - loss plot only accumulated loss
        - validation over 1000 images per epoch
        - smaller kernel size
        - smaller image crop
        - RAM???
"""
""" Install for Pytorch:
     
     1) must first install cuda
     2) conda install pytorch torchvision cudatoolkit=10.2 -c pytorch   # Must indicate cuda version???
     
     3) pip install torchio
    
     torch.__version__
     

     *** must add device(device) ==> to send things to GPU!

     
     ***crazy speed up with line below
     torch.backends.cudnn.benchmark = True
     
     
     torch.backends.cudnn.deterministic=True

     ***For uninstall:
     conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  
     
     
     put import statements at top of file or may crash???
     
     Add main function so can do multi-threading?
     no classes within main function
     
   
     use volatile == True for inference mode???
"""

""" ALLOWS print out of results on compute canada """
import matplotlib
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

""" Libraries to load """
import numpy as np
from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
import glob, os
import datetime
import time
#import bcolz
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from UNet_pytorch import *
from UNet_pytorch_online import *
from PYTORCH_dataloader import *

from sklearn.model_selection import train_test_split


# import kornia

from tracker import *

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    """" Input paths """    

    #s_path = '/media/user/FantomHD/Lightsheet data/Checkpoints_lightsheet/(1) Check_lightsheet_NO_transforms_AdamW_batch_norm/'
    #s_path = '/media/user/FantomHD/Lightsheet data/Checkpoints_lightsheet/(2) Check_lightsheet_NO_transforms_AdamW_batch_norm_5deep/'
    # s_path = '/media/user/FantomHD1/Lightsheet data/Checkpoints_lightsheet/(3) Check_lightsheet_YES_transforms_AdamW_batch_norm_5deep/'
    # s_path = '/media/user/FantomHD1/Lightsheet data/Checkpoints_lightsheet/(4) Check_lightsheet_NO_transforms_5deep_FIXED_SCALING_NEW_DATA/'
    # s_path = '/media/user/FantomHD1/Lightsheet data/Checkpoints_lightsheet/(5) Check_lightsheet_YES_transforms_5deep_FIXED_SCALING_NEW_DATA/'


    # s_path = '/media/user/FantomHD1/Lightsheet data/Checkpoints_lightsheet/(5) Check_lightsheet_YES_transforms_5deep_FIXED_SCALING_NEW_DATA/'

    # input_path = '/media/user/FantomHD1/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_blocks_128_16/'
    
    
    s_path = '/media/user/4TB_SSD/Training_UNet_Congo/(1) Check_lightsheet_NO_transforms_4deep_LR_same/'

    input_path = '/media/user/4TB_SSD/Training_UNet_Congo/Train_blocks_128_16_UNet/'


    resume = 0
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
    onlyfiles_check.sort(key = natsort_key1)
    
    
    # """ load mean and std """  
    mean_arr = np.load(input_path + 'normalize/mean_VERIFIED.npy')
    std_arr = np.load(input_path + 'normalize/std_VERIFIED.npy')       


    num_workers = 4;
    
    
    save_every_num_epochs = 1;
    plot_every_num_epochs = 1;
    validate_every_num_epochs = 1;  


    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_INPUT.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_INPUT.tif','_TRUTH.tif')) for i in images]
    counter = list(range(len(examples)))

    
    if not onlyfiles_check:   ### if no old checkpoints found, start new network and tracker
 
        """ Hyper-parameters """
        deep_sup = False
        switch_norm = False
        sp_weight_bool = 0
        # transforms = initialize_transforms(p=0.5)
        # transforms = initialize_transforms_simple(p=0.5)
        transforms = 0
        batch_size = 8;      
        test_size = 0.1  
        next_seg = 0;
        HD = 0;
        alpha = 0
        
        

        """ Initialize network """  
        kernel_size = 5
        pad = int((kernel_size - 1)/2)
        unet = UNet_online(in_channels=1, n_classes=2, depth=4, wf=3, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
                            batch_norm=True, batch_norm_switchable=switch_norm, up_mode='upsample')
        #unet = NestedUNet(num_classes=2, input_channels=2, deep_sup=deep_sup, padding=pad, batch_norm_switchable=switch_norm)
        #unet = UNet_3Plus(num_classes=2, input_channels=2, kernel_size=kernel_size, padding=pad)

        unet.train()
        unet.to(device)
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Select loss function *** unimportant if using HD loss """
        if not HD:    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        else:         loss_function = 'Haussdorf'
            

        """ Select optimizer """
        lr = 1e-5; milestones = [3000000, 100000000]  # with AdamW slow down
        optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

        """ Add scheduler """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
            
        """ initialize index of training set and validation set, split using size of test_size """
        
        idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
        """ initialize training_tracker """
        # idx_valid = idx_skip
        # idx_train = counter
        
        tracker = tracker(batch_size, test_size, mean_arr, std_arr, idx_train, idx_valid, deep_sup=deep_sup, switch_norm=switch_norm, alpha=alpha, HD=HD,
                                          sp_weight_bool=sp_weight_bool, transforms=transforms, dataset=input_path)

     
    else:             
        """ Find last checkpoint """       
        last_file = onlyfiles_check[-1]
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint

        print('restoring weights from: ' + checkpoint)
        check = torch.load(s_path + checkpoint, map_location=lambda storage, loc: storage)
        #check = torch.load(s_path + checkpoint, map_location='cpu')
        #check = torch.load(s_path + checkpoint, map_location=device)
        
        tracker = check['tracker']
        
        unet = check['model_type']
        optimizer = check['optimizer_type']
        scheduler = check['scheduler_type']
        unet.load_state_dict(check['model_state_dict'])
        unet.to(device)
        optimizer.load_state_dict(check['optimizer_state_dict'])
        scheduler.load_state_dict(check['scheduler'])     
        loss_function = check['loss_function']

        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Clean up checkpoint file """
        del check
        torch.cuda.empty_cache()
        
        
        

    """ 
        Time for 50 iterations (batch size 2)
        411.34 ==> 0 workers
        183.78438819999997 ==> 2 workers 
        127 ==> 4 workers ==> 60 secs training generator time
        111.03402260000001 ==> no transform
        109 ==> 6 workers ==> 90 secs training generator time   
        110.0 ==> 6 workers WITH DIRECT LOADING FROM BCOLZ ==> 30 secs generator tim
        max is 1000 samples loaded with batch size 2 and even with batch size 1??? 
        
        
        On Titan:
            time for 50 batch size 2 ==> 71 secs
            time for 50 batch size 4 ==> 116 secs (with transforms)
        
    """ 


    
    # if not resume:
    #     idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
    training_set = Dataset_tiffs(tracker.idx_train, examples, tracker.mean_arr, tracker.std_arr,
                                           sp_weight_bool=tracker.sp_weight_bool, transforms = tracker.transforms)
    val_set = Dataset_tiffs(tracker.idx_valid, examples, tracker.mean_arr, tracker.std_arr,
                                      sp_weight_bool=tracker.sp_weight_bool, transforms = 0)
    

    """ Create training and validation generators"""
    val_generator = data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers,
                    pin_memory=True, drop_last = True)

    training_generator = data.DataLoader(training_set, batch_size=tracker.batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True)
         
    print('Total # training images per epoch: ' + str(len(training_set)))
    print('Total # validation images: ' + str(len(val_set)))
    

    """ Epoch info """
    train_steps_per_epoch = len(tracker.idx_train)/tracker.batch_size
    validation_size = len(tracker.idx_valid)
    epoch_size = len(tracker.idx_train)    
   

    print('Total # training images per epoch: ' + str(len(training_set)))
    print('Total # validation images: ' + str(len(val_set)))

    """ Epoch info """
    train_steps_per_epoch = len(tracker.idx_train)/tracker.batch_size
    validation_size = len(tracker.idx_valid)
    epoch_size = len(tracker.idx_train) 

    zzz


    """ Find LR, to do so must: (1) have no workers, (2) uncomment section in dataloader """  
    # from torch_lr_finder import LRFinder

    # model = unet
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_finder.range_test(training_generator, end_lr=100, num_iter=100)
    # lr_finder.plot() # to inspect the loss-learning rate graph
    # #lr_finder.reset() # to reset the model and optimizer to their initial state
        
   

    """ Start training """
    starter = 0
    for cur_epoch in range(len(tracker.train_loss_per_epoch), 10000): 
         unet.train        
         loss_train = 0
         jacc_train = 0   
                  
         """ check and plot params during training """             
         for param_group in optimizer.param_groups:
               #tracker.alpha = 0.5
               #param_group['lr'] = 1e-6   # manually sets learning rate
               cur_lr = param_group['lr']
               tracker.lr_plot.append(cur_lr)
               tracker.print_essential()

              
              
         iter_cur_epoch = 0;          
         for batch_x, batch_y, spatial_weight in training_generator:
                starter += 1
                if starter == 1:
                    start = time.perf_counter()
                if starter == 50:
                    stop = time.perf_counter(); diff = stop - start; print(diff)

                """ Plot for debug """    
                # np_inputs = np.asarray(batch_x.numpy()[0], dtype=np.uint16)
                # np_labels = np.asarray(batch_y.numpy()[0], dtype=np.uint16)
                # np_labels[np_labels > 0] = 255
                
                # imsave(s_path + str(iter_cur_epoch) + '_input.tif', np_inputs)
                # imsave(s_path + str(iter_cur_epoch) + '_label.tif', np_labels)
                
                # in_max = plot_max(np_inputs, plot=0)
                # lb_max = plot_max(np_labels, plot=0)
                
                # imsave(s_path + str(iter_cur_epoch) + '_max_input.tif', in_max)
                # imsave(s_path + str(iter_cur_epoch) + '_max_label.tif', lb_max)            
                
                
                
                
                
                """ Load data ==> shape is (batch_size, num_channels, depth, height, width)
                     (1) converts to Tensor
                     (2) normalizes + appl other transforms on GPU
                     (3) ***add non-blocking???
                     ***INPUT LABELS MUST BE < 255??? or else get CudNN error
                     
                """
                inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)
   
                """ zero the parameter gradients"""
                optimizer.zero_grad()       
                
                """ forward + backward + optimize """
                output_train = unet(inputs)
                loss = loss_function(output_train, labels)
                if torch.is_tensor(spatial_weight):
                     spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                     weighted = loss * spatial_tensor
                     loss = torch.mean(weighted)
                else:
                     loss = torch.mean(loss)   
                     #loss
                
                
                loss.backward()
                optimizer.step()
               
                """ Training loss """
                tracker.train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                loss_train += loss.cpu().data.numpy()
                
   
                """ Calculate Jaccard on GPU """                 
                jacc = jacc_eval_GPU_torch(output_train, labels)
                jacc = jacc.cpu().data.numpy()
                                            
                jacc_train += jacc # Training jacc
                tracker.train_jacc_per_batch.append(jacc)
   
                tracker.iterations = tracker.iterations + 1       
                iter_cur_epoch += 1
                if tracker.iterations % 100 == 0:
                     print('Trained: %d' %(tracker.iterations))
                     
                     
                
                     
                     
               
                
    
         tracker.train_loss_per_epoch.append(loss_train/iter_cur_epoch)
         tracker.train_jacc_per_epoch.append(jacc_train/iter_cur_epoch)                       
    
         """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
         loss_val = 0; jacc_val = 0
         precision_val = 0; sensitivity_val = 0; val_idx = 0;
         iter_cur_epoch = 0;
         iter_total_val = 0;
         if cur_epoch % validate_every_num_epochs == 0:
             
              with torch.set_grad_enabled(False):  # saves GPU RAM
                  unet.eval()
                  for batch_x_val, batch_y_val, spatial_weight in val_generator:
                        
                        """ Transfer to GPU to normalize ect... """
                        inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
             
                        # forward pass to check validation
                        output_val = unet(inputs_val)
                        loss = loss_function(output_val, labels_val)
                        if torch.is_tensor(spatial_weight):
                               spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                               weighted = loss * spatial_tensor
                               loss = torch.mean(weighted)
                        else:
                               loss = torch.mean(loss)  
          
                        """ Training loss """
                        tracker.val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                        loss_val += loss.cpu().data.numpy()
                        
                        
                        ### skip if no cells found in ground truth
                        if len(np.where(batch_y_val[0] > 0)[0]) == 0:  
                            iter_total_val += 1
                            continue
                        
                        ### otherwise calculate jaccard, precision, sensitivity
                        else:
                                         
                            """ Calculate jaccard on GPU """
                            jacc = jacc_eval_GPU_torch(output_val, labels_val)
                            jacc = jacc.cpu().data.numpy()
                            
                            
                            jacc_val += jacc
                            tracker.val_jacc_per_batch.append(jacc)   
    
    
                            """ Convert back to cpu """                                      
                            output_val = output_val.cpu().data.numpy()            
                            output_val = np.moveaxis(output_val, 1, -1)
                            
                            """ Calculate sensitivity + precision as other metrics ==> only ever on ONE IMAGE of a batch"""
                            batch_x_val_cpu = batch_x_val.cpu().data.numpy() 
                             
                            batch_y_val_cpu = batch_y_val.cpu().data.numpy() 
                            seg_val = np.argmax(output_val[0], axis=-1)  
                            TP, FN, FP = find_TP_FP_FN_from_im(seg_val, batch_y_val_cpu[0])
                            
                            
    
                            
                            # if jacc < 0.05:
                            #     zzz
                            
                                           
                            if TP + FN == 0: TP; #print('skip');
                            else: 
                                sensitivity = TP/(TP + FN); sensitivity_val += sensitivity;    # PPV
                                iter_cur_epoch += 1  ### ONLY COUNT WHEN THERE ARE CELLS!!! Otherwise sensitivity == 0 for image
                                print(jacc)
                                
                            if TP + FP == 0: TP;
                            else: precision = TP/(TP + FP);  precision_val += precision    # precision             
                  
                            val_idx = val_idx + 1
                            print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                    
                            
                            # print(sensitivity)
                            # zzz
                            iter_total_val += 1
            
              

              tracker.val_loss_per_eval.append(loss_val/iter_total_val)
              tracker.val_jacc_per_eval.append(jacc_val/iter_cur_epoch)   
                   
              tracker.plot_prec.append(precision_val/iter_cur_epoch)
              tracker.plot_sens.append(sensitivity_val/iter_cur_epoch)
              
                   
                  
              """ Add to scheduler to do LR decay """
              scheduler.step()
                  
         if cur_epoch % plot_every_num_epochs == 0:       
              """ Plot sens + precision + jaccard + loss """
              plot_metric_fun(tracker.plot_sens, tracker.plot_sens_val, class_name='', metric_name='sensitivity', plot_num=30)
              plt.figure(30); plt.savefig(s_path + 'Sensitivity.png')
                    
              plot_metric_fun(tracker.plot_prec, tracker.plot_prec_val, class_name='', metric_name='precision', plot_num=31)
              plt.figure(31); plt.savefig(s_path + 'Precision.png')
           
              plot_metric_fun(tracker.train_jacc_per_epoch, tracker.val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
              plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
                   
                 
              plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
              plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          
                   
              
              plot_metric_fun(tracker.lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
              plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 
     

              """ Plot metrics per batch """                
              plot_metric_fun(tracker.train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
              plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
                                
              plot_cost_fun(tracker.train_loss_per_batch, tracker.train_loss_per_batch)                   
              plt.figure(18); plt.savefig(s_path + 'global_loss.png')
              plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
              plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
                                 
              plot_depth = 8
              output_train = output_train.cpu().data.numpy()            
              output_train = np.moveaxis(output_train, 1, -1)              
              seg_train = np.argmax(output_train[0], axis=-1)  
              
              # convert back to CPU
              batch_x = batch_x.cpu().data.numpy() 
              batch_y = batch_y.cpu().data.numpy() 
              batch_x_val = batch_x_val.cpu().data.numpy()
              
              plot_trainer_3D_PYTORCH(seg_train, seg_val, batch_x[0], batch_x_val_cpu[0], batch_y[0], batch_y_val_cpu[0],
                                       s_path, tracker.iterations, plot_depth=plot_depth)
                                             
              
         """ To save (every x iterations) """
         if cur_epoch % save_every_num_epochs == 0:                          

               save_name = s_path + 'check_' +  str(tracker.iterations)               
               torch.save({
                'tracker': tracker,

                'model_type': unet,
                'optimizer_type': optimizer,
                'scheduler_type': scheduler,
                
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss_function': loss_function,  
                
                }, save_name)
                

                
               
              
              