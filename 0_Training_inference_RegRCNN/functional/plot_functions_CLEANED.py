# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:16:39 2017

@author: Tiger
"""


import math
import pylab as mpl
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import random
#from skimage import measure





""" For plotting the output as an interactive scroller"""
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def plot_max(im):
     ma = np.amax(im, axis=0)
     plt.figure(); plt.imshow(ma)

     
     
""" Plots generic training outputs for Huganir lab data """
def plot_trainer_3D_PYTORCH(seg_train, seg_val, input_im, input_im_val, truth_im, truth_im_val,
                 s_path, epochs, plot_depth=0):
       """ Plot for debug """

       fig = plt.figure(num=3, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       # plt.subplot(3,2,1); plt.imshow(input_im[plot_depth, :, :]); plt.title('Input image train');
       # plt.subplot(3,2,3); plt.imshow(truth_im[plot_depth, :, :]); plt.title('Truth Train');
       # plt.subplot(3,2,5); plt.imshow(seg_train[plot_depth, :, :]); plt.title('Output Train');
       
       # plt.subplot(3,2,2); plt.imshow(input_im_val[plot_depth, :, :]); plt.title('Input image val');
       # plt.subplot(3,2,4); plt.imshow(truth_im_val[plot_depth, :, :]); plt.title('Truth val');
       # plt.subplot(3,2,6); plt.imshow(seg_val[plot_depth, :, :]); plt.title('Output val');   
       
       # plt.savefig(s_path + '_' + str(epochs) + '_output.png')
       
       
       """ Plot for max project evaluate """
       truth_im = np.amax(truth_im, axis= 0)
       truth_im_val = np.amax(truth_im_val, axis = 0)
       seg_train = np.amax(seg_train, axis = 0)
       seg_val = np.amax(seg_val, axis = 0)


       input_im = np.amax(input_im, axis = 0)
       input_im_val = np.amax(input_im_val, axis = 0)                                          


       fig = plt.figure(num=5, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(3,2,1); plt.imshow(input_im[:, :]); plt.title('Input image train');
       plt.subplot(3,2,3); plt.imshow(truth_im[:, :]); plt.title('Truth Train');
       plt.subplot(3,2,5); plt.imshow(seg_train[:, :]); plt.title('Output Train');
       
       plt.subplot(3,2,2); plt.imshow(input_im_val[:,  :]); plt.title('Input image val');
       plt.subplot(3,2,4); plt.imshow(truth_im_val[:, :]); plt.title('Truth val');
       plt.subplot(3,2,6); plt.imshow(seg_val[:, :]); plt.title('Output val');   
       
       plt.savefig(s_path + '_' + str(epochs) + '_max_project_output.png') 
       

""" Plots generic training outputs for Huganir lab data """
def plot_trainer_2D_PYTORCH(seg_train, seg_val, input_im, input_im_val, truth_im, truth_im_val,
                 s_path, epochs, plot_depth=0):
       """ Plot for debug """

       fig = plt.figure(num=3, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(3,2,1); plt.imshow(input_im); plt.title('Input image train');
       plt.subplot(3,2,3); plt.imshow(truth_im); plt.title('Truth Train');
       plt.subplot(3,2,5); plt.imshow(seg_train); plt.title('Output Train');
       
       plt.subplot(3,2,2); plt.imshow(input_im_val); plt.title('Input image val');
       plt.subplot(3,2,4); plt.imshow(truth_im_val); plt.title('Truth val');
       plt.subplot(3,2,6); plt.imshow(seg_val); plt.title('Output val');   
       
       plt.savefig(s_path + '_' + str(epochs) + '_output.png')
       
       
       """ Plot for max project evaluate """
       # truth_im = np.amax(truth_im, axis= 0)
       # truth_im_val = np.amax(truth_im_val, axis = 0)
       # seg_train = np.amax(seg_train, axis = 0)
       # seg_val = np.amax(seg_val, axis = 0)


       # input_im = np.amax(input_im, axis = 0)
       # input_im_val = np.amax(input_im_val, axis = 0)                                          


       # fig = plt.figure(num=5, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       # fig.subplots_adjust(hspace=.3)
       
       # plt.subplot(3,2,1); plt.imshow(input_im[:, :]); plt.title('Input image train');
       # plt.subplot(3,2,3); plt.imshow(truth_im[:, :]); plt.title('Truth Train');
       # plt.subplot(3,2,5); plt.imshow(seg_train[:, :]); plt.title('Output Train');
       
       # plt.subplot(3,2,2); plt.imshow(input_im_val[:,  :]); plt.title('Input image val');
       # plt.subplot(3,2,4); plt.imshow(truth_im_val[:, :]); plt.title('Truth val');
       # plt.subplot(3,2,6); plt.imshow(seg_val[:, :]); plt.title('Output val');   
       
       # plt.savefig(s_path + '_' + str(epochs) + '_max_project_output.png') 



""" Plots generic training outputs for Huganir lab data """
def plot_trainer_3D_PYTORCH_snake_seg(seg_train, seg_val, input_im, input_im_val, truth_im, truth_im_val,
                 s_path, epochs, plot_depth=0):
    
       
       seed = input_im[1]
       input_im = input_im[0]
       
       seed_val = input_im_val[1]
       input_im_val = input_im_val[0]
    
    
       """ Plot for debug """

       fig = plt.figure(num=3, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(4,2,1); plt.imshow(input_im[plot_depth, :, :]); plt.title('Input image train');
       plt.subplot(4,2,3); plt.imshow(truth_im[plot_depth, :, :]); plt.title('Truth Train');
       plt.subplot(4,2,5); plt.imshow(seg_train[plot_depth, :, :]); plt.title('Output Train');
       plt.subplot(4,2,7); plt.imshow(seed[plot_depth, :, :]); plt.title('Seed Train');
       
       
       plt.subplot(4,2,2); plt.imshow(input_im_val[plot_depth, :, :]); plt.title('Input image val');
       plt.subplot(4,2,4); plt.imshow(truth_im_val[plot_depth, :, :]); plt.title('Truth val');
       plt.subplot(4,2,6); plt.imshow(seg_val[plot_depth, :, :]); plt.title('Output val');   
       plt.subplot(4,2,8); plt.imshow(seed_val[plot_depth, :, :]); plt.title('Seed val');   
       
       
       plt.savefig(s_path + '_' + str(epochs) + '_output.png')
       
       
       """ Plot for max project evaluate """
       truth_im = np.amax(truth_im, axis= 0)
       truth_im_val = np.amax(truth_im_val, axis = 0)
       seg_train = np.amax(seg_train, axis = 0)
       seg_val = np.amax(seg_val, axis = 0)


       input_im = np.amax(input_im, axis = 0)
       seed = np.amax(seed, axis=0)
       
       input_im_val = np.amax(input_im_val, axis = 0)                                          
       seed_val = np.amax(seed_val, axis=0)

       fig = plt.figure(num=5, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(4,2,1); plt.imshow(input_im[:, :]); plt.title('Input image train');
       plt.subplot(4,2,3); plt.imshow(truth_im[:, :]); plt.title('Truth Train');
       plt.subplot(4,2,5); plt.imshow(seg_train[:, :]); plt.title('Output Train');
       plt.subplot(4,2,7); plt.imshow(seed[:, :]); plt.title('Seed Train');
       
       
       plt.subplot(4,2,2); plt.imshow(input_im_val[:,  :]); plt.title('Input image val');
       plt.subplot(4,2,4); plt.imshow(truth_im_val[:, :]); plt.title('Truth val');
       plt.subplot(4,2,6); plt.imshow(seg_val[:, :]); plt.title('Output val');   
       plt.subplot(4,2,8); plt.imshow(seed_val[:, :]); plt.title('Seed val'); 
       
       plt.savefig(s_path + '_' + str(epochs) + '_max_project_output.png') 







""" Plots generic training outputs for Huganir lab data """
def plot_trainer_3D_HUGANIR(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, input_im, input_im_val, truth_im, truth_im_val,
                 s_path, epochs, plot_depth=0, multiclass=0):
       """ Plot for debug """
       feed_dict = feed_dict_TRAIN
       output_train = softMaxed.eval(feed_dict=feed_dict)
       seg_train = np.argmax(output_train, axis = -1)             
       
    
       feed_dict = feed_dict_CROSSVAL
       output_val = softMaxed.eval(feed_dict=feed_dict)
       seg_val = np.argmax(output_val, axis = -1)                
       

       
       fig = plt.figure(num=3, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(3,2,1); plt.imshow(input_im[plot_depth, :, :, 0]); plt.title('Input image train');
       plt.subplot(3,2,3); plt.imshow(truth_im[plot_depth, :, :, 1]); plt.title('Truth Train');
       plt.subplot(3,2,5); plt.imshow(seg_train[-1, plot_depth, :, :]); plt.title('Output Train');
       
       plt.subplot(3,2,2); plt.imshow(input_im_val[plot_depth, :, :, 0]); plt.title('Input image val');
       plt.subplot(3,2,4); plt.imshow(truth_im_val[plot_depth, :, :, 1]); plt.title('Truth val');
       plt.subplot(3,2,6); plt.imshow(seg_val[-1, plot_depth, :, :]); plt.title('Output val');   
       
       plt.savefig(s_path + '_' + str(epochs) + '_output.png')
       
       
       """ Plot for max project evaluate """
       truth_im = np.amax(truth_im, axis= 0)
       truth_im_val = np.amax(truth_im_val, axis = 0)
       seg_train = np.amax(seg_train[-1], axis = 0)
       seg_val = np.amax(seg_val[-1], axis = 0)
       
       
       #raw_truth = np.amax(raw_truth, axis = 0)
       #raw_truth_val = np.amax(raw_truth_val, axis = 0)
       input_im = np.amax(input_im, axis = 0)
       input_im_val = np.amax(input_im_val, axis = 0)                                          


       fig = plt.figure(num=5, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(3,2,1); plt.imshow(input_im[:, :, 0]); plt.title('Input image train');
       plt.subplot(3,2,3); plt.imshow(truth_im[:, :, 1]); plt.title('Truth Train');
       plt.subplot(3,2,5); plt.imshow(seg_train[:, :]); plt.title('Output Train');
       
       plt.subplot(3,2,2); plt.imshow(input_im_val[:,  :, 0]); plt.title('Input image val');
       plt.subplot(3,2,4); plt.imshow(truth_im_val[:, :, 1]); plt.title('Truth val');
       plt.subplot(3,2,6); plt.imshow(seg_val[:, :]); plt.title('Output val');   
       
       plt.savefig(s_path + '_' + str(epochs) + '_max_project_output.png') 
       
       
       
""" Plots generic training outputs """
def plot_trainer_3D(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, input_im, input_im_val, truth_im, truth_im_val,
                 weighted_labels, weighted_labels_val, s_path, epochs, plot_depth=0, multiclass=0):
       """ Plot for debug """
       feed_dict = feed_dict_TRAIN
       output_train = softMaxed.eval(feed_dict=feed_dict)
       seg_train = np.argmax(output_train, axis = -1)[-1]              
       
    
       feed_dict = feed_dict_CROSSVAL
       output_val = softMaxed.eval(feed_dict=feed_dict)
       seg_val = np.argmax(output_val, axis = -1)[-1]                  
       
       raw_truth = np.copy(truth_im)
       raw_truth_val = np.copy(truth_im_val)
       
       
       truth_im = np.argmax(truth_im, axis = -1)
       truth_im_val = np.argmax(truth_im_val, axis = -1)
       
       fig = plt.figure(num=3, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(4,2,1); plt.imshow(input_im[plot_depth, :, :, 0]); plt.title('Input image train');
       plt.subplot(4,2,3); plt.imshow(input_im[plot_depth, :, :, 1]); plt.title('Input image train seed');
       plt.subplot(4,2,5); plt.imshow(truth_im[plot_depth, :, :]); plt.title('Truth Train');
       plt.subplot(4,2,7); plt.imshow(seg_train[plot_depth, :, :]); plt.title('Output Train');
       
       plt.subplot(4,2,2); plt.imshow(input_im_val[plot_depth, :, :, 0]); plt.title('Input image val');
       plt.subplot(4,2,4); plt.imshow(input_im_val[plot_depth, :, :, 1]); plt.title('Input image val seed');
       plt.subplot(4,2,6); plt.imshow(truth_im_val[plot_depth, :, :]); plt.title('Truth val');
       plt.subplot(4,2,8); plt.imshow(seg_val[plot_depth, :, :]); plt.title('Output val');   
       
       plt.savefig(s_path + '_' + str(epochs) + '_output.png')
       
       
       """ Then plot all class info """
       fig = plt.figure(num=4, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       plt.subplot(4,2,3); plt.imshow(raw_truth[plot_depth, :, :, 1]); plt.title('Truth class 1: train');
       plt.subplot(4,2,5); plt.imshow(raw_truth[plot_depth, :, :, 0]); plt.title('Truth background');  
     
       plt.subplot(4,2,4); plt.imshow(raw_truth_val[plot_depth, :, :, 1]); plt.title('Truth class 1: val');
       plt.subplot(4,2,6); plt.imshow(raw_truth_val[plot_depth, :, :, 0]); plt.title('Truth background');  
 
       if multiclass:
            plt.subplot(4,2,1); plt.imshow(raw_truth[plot_depth, :, :, 2]); plt.title('Truth class 2: train');    #plt.pause(0.005)
            plt.subplot(4,2,2); plt.imshow(raw_truth_val[plot_depth, :, :, 2]); plt.title('Truth class 2: train');    #plt.pause(0.005)

      
       weighted_labels = np.amax(weighted_labels, axis = 0)
       weighted_labels_val = np.amax(weighted_labels_val, axis = 0)
       plt.subplot(4,2,7); plt.imshow(weighted_labels[:, :, 1]); plt.title('S_weight 1: train');
       plt.subplot(4,2,8); plt.imshow(weighted_labels_val[:, :, 1]); plt.title('S_weight 2: val');         
          
          
          
          
       plt.savefig(s_path + '_' + str(epochs) + '_output_class.png')
       
       
       
       
       """ Plot for max project evaluate """
       truth_im = np.amax(truth_im, axis= 0)
       truth_im_val = np.amax(truth_im_val, axis = 0)
       seg_train = np.amax(seg_train, axis = 0)
       seg_val = np.amax(seg_val, axis = 0)
       
       

       raw_truth = np.amax(raw_truth, axis = 0)
       raw_truth_val = np.amax(raw_truth_val, axis = 0)
       input_im = np.amax(input_im, axis = 0)
       input_im_val = np.amax(input_im_val, axis = 0)                                          
       


       fig = plt.figure(num=5, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(4,2,1); plt.imshow(input_im[:, :, 0]); plt.title('Input image train');
       plt.subplot(4,2,3); plt.imshow(input_im[:, :, 1]); plt.title('Input image train seed');
       plt.subplot(4,2,5); plt.imshow(truth_im[:, :]); plt.title('Truth Train');
       plt.subplot(4,2,7); plt.imshow(seg_train[:, :]); plt.title('Output Train');
       
       plt.subplot(4,2,2); plt.imshow(input_im_val[:,  :, 0]); plt.title('Input image val');
       plt.subplot(4,2,4); plt.imshow(input_im_val[:, :, 1]); plt.title('Input image val seed');
       plt.subplot(4,2,6); plt.imshow(truth_im_val[:, :]); plt.title('Truth val');
       plt.subplot(4,2,8); plt.imshow(seg_val[:, :]); plt.title('Output val');   
       
       plt.savefig(s_path + '_' + str(epochs) + '_max_project_output.png')
       
       
       """ Then plot all class info """
       fig = plt.figure(num=6, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       plt.subplot(4,2,1); plt.imshow(weighted_labels[:, :, 1]); plt.title('weighted train');    #plt.pause(0.005)
       plt.subplot(4,2,3); plt.imshow(raw_truth[:, :, 1]); plt.title('Truth class 1: train');
     
       plt.subplot(4,2,2); plt.imshow(weighted_labels_val[:, :, 1]); plt.title('weighted val');    #plt.pause(0.005)
       plt.subplot(4,2,4); plt.imshow(raw_truth_val[:, :, 1]); plt.title('Truth class 1: val');


       if multiclass:     
            plt.subplot(4,2,5); plt.imshow(raw_truth[:, :, 2]); plt.title('Truth class 2: train');    #plt.pause(0.005)       
            plt.subplot(4,2,6); plt.imshow(raw_truth_val[:, :, 2]); plt.title('Truth class 2: train');    #plt.pause(0.005)
       
       
       
"""
    Scales the normalized images to be within [0, 1], thus allowing it to be displayed
"""
def show_norm(im):
    m,M = im.min(),im.max()
    plt.imshow((im - m) / (M - m))
    plt.show()



  
""" Plots global and detailed cost functions
""" 
def plot_cost_fun(plot_cost, plot_cost_val, mov_avg_loss=None, plot_cost_val_NO=None):
      """ Graph global loss
      """      
      avg_window_size = 40
 
      if len(plot_cost) <= avg_window_size:
          mov_avg_loss = plot_cost
          mov_avg_loss_val = plot_cost_val
      else:
          mov_avg_loss = plot_cost[0:avg_window_size]    
          mov_avg_loss_val = plot_cost_val[0:avg_window_size]
     
          avg_loss = moving_average(plot_cost, n=avg_window_size).tolist()
          avg_loss_val = moving_average(plot_cost_val, n=avg_window_size).tolist()
          
          mov_avg_loss = mov_avg_loss + avg_loss
          mov_avg_loss_val = mov_avg_loss_val + avg_loss_val


      plt.figure(18); plt.clf();
      plt.plot(plot_cost, alpha=0.3, label='Training'); plt.title('Global Loss')
      if mov_avg_loss is not None:
           plt.plot(mov_avg_loss, color='tab:blue');
      plt.ylabel('Loss'); plt.xlabel('Iterations'); #plt.pause(0.05)
      plt.yscale('log')
      
      # cross-validation
      plt.figure(18); plt.plot(plot_cost_val, alpha=0.3, label='Validation'); #plt.pause(0.05)
      if mov_avg_loss_val is not None:
           plt.plot(mov_avg_loss_val, color='tab:orange');
      plt.legend(loc='upper left');    
      plt.yscale('log')


      plt.figure(25); plt.clf();
      plt.plot(plot_cost, alpha=0.3, label='Training'); plt.title('Global Loss')
      if mov_avg_loss is not None:
           plt.plot(mov_avg_loss, color='tab:blue');
      plt.ylabel('Loss'); plt.xlabel('Iterations'); #plt.pause(0.05)
      plt.xscale('log')
      plt.yscale('log')
      
      # cross-validation
      plt.figure(25); plt.plot(plot_cost_val, alpha=0.3, label='Validation'); #plt.pause(0.05)
      if mov_avg_loss_val is not None:
           plt.plot(mov_avg_loss_val, color='tab:orange');
      plt.legend(loc='upper left');    
      plt.xscale('log')
      plt.yscale('log')
      
      
      """ Graph detailed plot
      """
      last_loss = len(plot_cost)
      start = 0
      if last_loss < 50:
          start = 0
      elif last_loss < 200:
          start = last_loss - 50
          
      elif last_loss < 500:
          start = last_loss - 200
          
      elif last_loss < 1500:
          start = last_loss - 500
          
      elif last_loss < 10000:
          start = last_loss - 1500 
      else:
          start = last_loss - 8000
      plt.close(19);
      x_idx = list(range(start, last_loss))
      plt.figure(19); plt.plot(x_idx,plot_cost[start:last_loss], alpha=0.3, label='Training'); plt.title("Detailed Loss"); 
      plt.figure(19); plt.plot(x_idx,plot_cost_val[start:last_loss], alpha=0.3, label='Validation');
      plt.legend(loc='upper left');             
      plt.ylabel('Loss'); plt.xlabel('Iterations'); #plt.pause(0.05) 
      #plt.xscale('log')
      plt.yscale('log')

      
      if plot_cost_val_NO is not None:
            plt.figure(18); plt.plot(plot_cost_val_NO, label='Cross_validation_NO'); #plt.pause(0.05)                                      
            plt.figure(19); plt.plot(x_idx, plot_cost_val_NO[start:last_loss], label='Validation_NO');   #plt.pause(0.05)    
      

      
     
""" Plots any metric with moving average
"""
def plot_metric_fun(plot_jaccard, plot_jaccard_val=False, class_name='', metric_name='Jaccard', plot_num=30):
      
     avg_window_size = 40
 
     if len(plot_jaccard) <= avg_window_size:
          mov_avg_jacc = plot_jaccard
          mov_avg_jacc_val = plot_jaccard_val
     else:
          mov_avg_jacc = plot_jaccard[0:avg_window_size]    
          mov_avg_jacc_val = plot_jaccard_val[0:avg_window_size]
     
          avg_jacc = moving_average(plot_jaccard, n=avg_window_size).tolist()
          avg_jacc_val = moving_average(plot_jaccard_val, n=avg_window_size).tolist()
          
          mov_avg_jacc = mov_avg_jacc + avg_jacc
          mov_avg_jacc_val = mov_avg_jacc_val + avg_jacc_val
                

     """ Graph global metric
     """      
     plt.figure(plot_num); plt.clf();
     plt.plot(plot_jaccard, alpha=0.3, label=metric_name + class_name); plt.title(metric_name)  
     plt.plot(mov_avg_jacc, color='tab:blue');

     if plot_jaccard_val:
         plt.plot(plot_jaccard_val, alpha=0.3, label='Validation ' + metric_name + ' ' + class_name);
         plt.plot(mov_avg_jacc_val, color='tab:orange');

     plt.ylabel(metric_name); plt.xlabel('Epochs');            
     plt.legend(loc='upper left');    #plt.pause(0.05)
      


""" Easier moving average calculation """
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
  



""" Originally from Intro_to_deep_learning workshop
"""
def plotOutput(layer,feed_dict,fieldShape=None,channel=None,figOffset=1,cmap=None):
	# Output summary
	W = layer
	wp = W.eval(feed_dict=feed_dict);
	if len(np.shape(wp)) < 4:		# Fully connected layer, has no shape
		temp = np.zeros(np.product(fieldShape)); temp[0:np.shape(wp.ravel())[0]] = wp.ravel()
		fields = np.reshape(temp,[1]+fieldShape)
	else:			# Convolutional layer already has shape
		wp = np.rollaxis(wp,3,0)
		features, channels, iy,ix = np.shape(wp)   # where "features" is the number of "filters"
		if channel is not None:
			fields = wp[:,channel,:,:]
		else:
			fields = np.reshape(wp,[features*channels,iy,ix])    # all to remove "channels" axis

	perRow = int(math.floor(math.sqrt(fields.shape[0])))
	perColumn = int(math.ceil(fields.shape[0]/float(perRow)))
	fields2 = np.vstack([fields,np.zeros([perRow*perColumn-fields.shape[0]] + list(fields.shape[1:]))])    # adds more zero filters...
	tiled = []
	for i in range(0,perColumn*perRow,perColumn):
		tiled.append(np.hstack(fields2[i:i+perColumn]))    # stacks horizontally together ALL the filters

	tiled = np.vstack(tiled)    # then stacks itself on itself
	if figOffset is not None:
		mpl.figure(figOffset); mpl.clf(); 

	mpl.imshow(tiled,cmap=cmap); mpl.title('%s Output' % layer.name); mpl.colorbar();
    
    
""" Plot layers
"""
def plotLayers(feed_dict, L1, L2, L3, L4, L5, L6, L8, L9, L10):
      plt.figure('Down_Layers');
      plt.clf()
      plt.subplot(221); plotOutput(L1,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(222); plotOutput(L2,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(233); plotOutput(L3,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(234); plotOutput(L5,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(223); plotOutput(L4,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.pause(0.05)
      
      plt.figure('Up_Layers');
      plt.clf()
      plt.subplot(221); plotOutput(L6,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(222); plotOutput(L8,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(223); plotOutput(L9,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(224); plotOutput(L10,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.pause(0.05); 
      
 
    