#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:39:11 2020

@author: user
"""
from plot_functions_CLEANED import *
import matplotlib.pyplot as plt


""" Tracker class """
class tracker():
    
    def __init__(self, batch_size, test_size, mean_arr, std_arr, idx_train, idx_valid, deep_sup, switch_norm, alpha=1, HD=1, sp_weight_bool=0, transforms=0, dataset=''):
        """ Get metrics per batch """
        self.train_loss_per_batch = [] 
        self.train_jacc_per_batch = []
        self.val_loss_per_batch = []; self.val_jacc_per_batch = []
        
        self.train_ce_pb = []; self.train_hd_pb = []; self.train_dc_pb = [];
        self.val_ce_pb = []; self.val_hd_pb = []; self.val_dc_pb = [];
 
        """ Get metrics per epoch"""
        self.train_loss_per_epoch = []; self.train_jacc_per_epoch = []
        self.val_loss_per_eval = []; self.val_jacc_per_eval = []
        self. plot_sens = []; self.plot_sens_val = [];
        self.plot_prec = []; self.plot_prec_val = [];
        self.lr_plot = [];
        self.iterations = 0;
        self.cur_epoch = 0;
        
        self.resize_z = 0;


        """ Normalization """
        self.mean_arr = mean_arr
        self.std_arr = std_arr
        

        """ save index of training data """
        self.idx_train = idx_train
        self.idx_valid = idx_valid
        
        """ Netwrok params """
        self.batch_size = batch_size
        self.test_size = test_size
        
        self.deep_sup = deep_sup
        self.switch_norm = switch_norm 
        self.alpha = alpha
        self.HD = HD
        self.sp_weight_bool = sp_weight_bool
        self.transforms = transforms
        self.dataset = dataset


    def print_essential(self):
        print('Current learning rate is: ' + str(self.lr_plot[-1]))
        print('Weight bool is: ' + str(self.sp_weight_bool))
        print('switch norm bool is: ' + str(self.switch_norm))
        print('batch_size is: ' + str(self.batch_size))
        print('dataset is: ' + self.dataset)              
        print('deep_supervision is: ' + str(self.deep_sup))
        print('alpha is: ' + str(self.alpha))    
        print('HD_bool is: ' + str(self.HD))    
        

def show_vars(obj_name):
    params = [attr for attr in dir(obj_name) if not callable(getattr(obj_name, attr)) and not attr.startswith("__")]
    print(params)
    return


""" Plot metrics in tracker """           
def plot_tracker(tracker, s_path):

    plot_metric_fun(tracker.train_jacc_per_epoch, tracker.val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
    plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
    
       
    plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
    plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          
    
    
    plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=40)
    plt.figure(40); plt.savefig(s_path + 'loss_per_epoch_NO_LOG.png')         
     
    
      
    """ Separate losses """
    if tracker.HD:
        plot_cost_fun(tracker.train_ce_pb, tracker.train_ce_pb)                   
        plt.figure(18); plt.savefig(s_path + '_global_loss_CE.png')
        #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
        plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_CE.png')
        plt.close('all')
          
        plot_cost_fun(tracker.train_hd_pb, tracker.train_hd_pb)                   
        plt.figure(18); plt.savefig(s_path + '_global_loss_HD.png')
        #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
        plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_HD.png')
        plt.close('all')
          
        plot_cost_fun(tracker.train_dc_pb, tracker.train_dc_pb)                   
        plt.figure(18); plt.savefig(s_path + '_global_loss_DC.png')
        #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
        plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_DC.png')
        plt.close('all')                  
        
 
        ### for validation
        plot_cost_fun(tracker.val_ce_pb, tracker.val_ce_pb)                   
        plt.figure(18); plt.savefig(s_path + '_VAL_global_loss_CE.png')
        #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
        plt.figure(25); plt.savefig(s_path + '_VAL_global_loss_LOG_CE.png')
        plt.close('all')
          
        plot_cost_fun(tracker.val_hd_pb, tracker.val_hd_pb)                   
        plt.figure(18); plt.savefig(s_path + '_VAL_global_loss_HD.png')
        #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
        plt.figure(25); plt.savefig(s_path + '_VAL_global_loss_LOG_HD.png')
        plt.close('all')
          
        plot_cost_fun(tracker.val_dc_pb, tracker.val_dc_pb)                   
        plt.figure(18); plt.savefig(s_path + '_VAL_global_loss_DC.png')
        #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
        plt.figure(25); plt.savefig(s_path + '_VAL_global_loss_LOG_DC.png')
        plt.close('all')        
    
 
    
    """ VALIDATION LOSS PER BATCH??? """
    plot_cost_fun(tracker.val_loss_per_batch, tracker.val_loss_per_batch)                   
    plt.figure(18); plt.savefig(s_path + '_VAL_global_loss_VAL.png')
    plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
    plt.figure(25); plt.savefig(s_path + '_VAL_global_loss_LOG_VAL.png')
    plt.close('all')
      
    
    plot_metric_fun(tracker.lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
    plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 
    
    """ Plot metrics per batch """                
    plot_metric_fun(tracker.train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
    plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
      
    plot_cost_fun(tracker.train_loss_per_batch, tracker.train_loss_per_batch)                   
    plt.figure(18); plt.savefig(s_path + 'global_loss.png')
    plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
    plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
    plt.close('all')




