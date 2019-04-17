import tensorflow as tf
import numpy as np
import cv2 as cv
import random
import math
from sklearn.utils import shuffle
import pdb
import re
import data_reader as reader

import networks as nets
import utils
import params

SHOW_IMAGES = False 
IS_RESTORE = tf.train.latest_checkpoint(params.folder_data) != None 
 
params.show_params()   
data_reader = reader.DataReader('./data/train', './data/validation', './data/test')
   	 
# training  
dim_h_w = int(data_reader.dim_patch / params.scale)
dim_depth = int(data_reader.dim_depth / params.scale)

# resize heigth and width
input = tf.placeholder(tf.float32, (dim_depth, dim_h_w, dim_h_w, params.num_channels), name='input') 
target_h_w = tf.placeholder(tf.float32, (dim_depth, data_reader.dim_patch, data_reader.dim_patch, params.num_channels), name='target_h_w') 
output_h_w = params.network_architecture_H_W(input, params.kernel_size)
 
target_d = tf.placeholder(tf.float32, (data_reader.dim_depth, data_reader.dim_patch, data_reader.dim_patch, params.num_channels), name='target_d')  
output_before = params.network_architecture_D(tf.transpose(target_h_w, [1, 2, 0, 3]), params.kernel_size) 

output_d = tf.transpose(output_before, [2, 0, 1, 3])
  
if(params.LOSS == params.L1_LOSS): # 
	loss_h_w = tf.reduce_mean(tf.abs(output_h_w - target_h_w))
	loss_d = tf.reduce_mean(tf.abs(output_d - target_d))
if(params.LOSS == params.L2_LOSS):
	loss_h_w = tf.reduce_mean(tf.square(output_h_w - target_h_w))  
	loss_d = tf.reduce_mean(tf.square(output_d - target_d))
	   
opt_h_w = tf.train.AdamOptimizer(params.learning_rate_h_w).minimize(loss_h_w) 
opt_d = tf.train.AdamOptimizer(params.learning_rate_d).minimize(loss_d)

config = tf.ConfigProto(
        device_count = {'GPU': 1}
    ) 
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer()) 
  
writer = tf.summary.FileWriter('train.log', sess.graph)
  
saver = tf.train.Saver(max_to_keep=0)   
start_epoch = 0

if(IS_RESTORE):
    print('===========restoring from ' + tf.train.latest_checkpoint(params.folder_data))
    saver.restore(sess,tf.train.latest_checkpoint(params.folder_data))
    start_epoch = re.findall(r'\d+', tf.train.latest_checkpoint(params.folder_data))
    start_epoch = int(start_epoch[0]) + 1 
    
for epoch in range(start_epoch, params.num_epochs):
	   
	num_images = 0 
	num_iterations = data_reader.num_train_images  
	ssim_epoch_h_w = 0
	psnr_epoch_h_w = 0
	batch_loss_h_w = 0
	ssim_epoch_d = 0
	psnr_epoch_d = 0 
	batch_loss_d = 0
	for i in range(0, num_iterations): 
		 input_h_w_, target_d_, target_h_w_  = data_reader.get_next_batch_train(i) 
		 num_images += 1 
		 cost_h_w, cost_d, output_h_w_, output_d_, _, _= sess.run([loss_h_w, loss_d, output_h_w, output_d, opt_h_w, opt_d], feed_dict={input: input_h_w_ , target_h_w: target_h_w_, target_d: target_d_})    
         
		 cv.imshow('i', input_h_w_[0] / 255)
		 cv.imshow('output_h_w_', output_h_w_[0] / 255)
		 cv.imshow('target_h_w_', target_h_w_[0] / 255)
		 cv.imshow('output_d_', output_d_[0] / 255)
		 cv.imshow('target_d_', target_d_[0] / 255)
		 cv.waitKey(1000)
         
         
		 batch_loss_h_w += cost_h_w * target_h_w_.shape[0] 
		 ssim_batch, psnr_batch = utils.compute_ssim_psnr_batch(target_h_w_, output_h_w_)
		 ssim_epoch_h_w += ssim_batch / target_h_w_.shape[0]
		 psnr_epoch_h_w += psnr_batch /target_h_w_.shape[0]
         
		 batch_loss_d += cost_d * target_d_.shape[0]
		 ssim_batch, psnr_batch = utils.compute_ssim_psnr_batch(output_d_, target_d_)
		 ssim_epoch_d += ssim_batch / target_d_.shape[0]
		 psnr_epoch_d += psnr_batch / target_d_.shape[0]
		 print("Epoch/Iteration/Global Iteration: {}/{} ...".format(epoch, i), "Training loss_h_w: {:.8f} loss_d {:.8f}".format(batch_loss_h_w / (num_images * dim_depth), batch_loss_d / (num_images * params.dim_depth)), "Learning rate_w, lr_d:  {:.8f}".format(params.learning_rate_h_w, params.learning_rate_h_w))  
	
	tf.summary.scalar('loss_h_w', batch_loss_h_w / (num_images * dim_depth)) 
	tf.summary.scalar('learning_rate_h_w', params.learning_rate_h_w)  
	tf.summary.scalar('ssim_h_w', ssim_epoch_h_w / (num_images * dim_depth))  
	tf.summary.scalar('psnr_h_w', psnr_epoch_h_w / (num_images * dim_depth))  
	tf.summary.scalar('loss_d', batch_loss_d / (num_images * params.dim_depth)) 
	tf.summary.scalar('learning_rate_d', params.learning_rate_d)  
	tf.summary.scalar('ssim_d', ssim_epoch_d / (num_images * params.dim_depth))  
	tf.summary.scalar('psnr_d', psnr_epoch_d / (num_images * params.dim_depth))  
	merged = tf.summary.merge_all()
    
	merged_ = sess.run(merged)
	writer.add_summary(merged_, epoch)
	if(epoch % 50 == 0):
		 print('saving checkpoint...')  
		 saver.save(sess, params.folder_data + params.ckpt_name + str(epoch))	

sess.close()