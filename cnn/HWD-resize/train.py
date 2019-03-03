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

input = tf.placeholder(tf.float32, (dim_depth, dim_h_w, dim_h_w, params.num_channels), name='input')
target = tf.placeholder(tf.float32, (data_reader.dim_depth, data_reader.dim_patch, data_reader.dim_patch, params.num_channels), name='target')
target_h_w = tf.placeholder(tf.float32, (dim_depth, data_reader.dim_patch, data_reader.dim_patch, params.num_channels), name='target_h_w')

output_h_w_before, features_h_w = params.network_architecture_H_W(input, params.kernel_size) 
output_h_w = tf.transpose(output_h_w_before, [1, 2, 0, 3])

output_before, features, after_ps = params.network_architecture_D(output_h_w, params.kernel_size) 

output = tf.transpose(output_before, [2, 0, 1, 3])
 
print('output shape is ', output.shape)
if(params.LOSS == params.L1_LOSS): # 
	loss = tf.reduce_mean(tf.abs(output - target)) + tf.reduce_mean(tf.abs(output_h_w_before - target_h_w))
if(params.LOSS == params.L2_LOSS):
	loss = tf.reduce_mean(tf.square(output - target)) + tf.reduce_mean(tf.square(output_h_w_before - target_h_w))
	 
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = params.learning_rate 
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           500, 0.75, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

config = tf.ConfigProto(
        device_count = {'GPU': 1}
    ) 
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer()) 

total_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="total_loss")
lr_placeholder = tf.placeholder(tf.float32, shape=[], name="total_loss")
ssim_placeholder = tf.placeholder(tf.float32, shape=[], name="ssim_placeholder")
psnr_placeholder = tf.placeholder(tf.float32, shape=[], name="psnr_placeholder")
tf.summary.scalar('loss', total_loss_placeholder) 
tf.summary.scalar('learning_rate', lr_placeholder)  
tf.summary.scalar('ssim', ssim_placeholder)  
tf.summary.scalar('psnr', psnr_placeholder)  
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('train.log', sess.graph)
  
saver = tf.train.Saver(max_to_keep=0)   
start_epoch = 0

if(IS_RESTORE):
    print('===========restoring from ' + tf.train.latest_checkpoint(params.folder_data))
    saver.restore(sess,tf.train.latest_checkpoint(params.folder_data))
    start_epoch = re.findall(r'\d+', tf.train.latest_checkpoint(params.folder_data))
    start_epoch = int(start_epoch[0]) + 1 
    
for epoch in range(start_epoch, params.num_epochs):
	batch_loss = 0   
	num_images = 0 
	num_iterations = data_reader.num_train_images  
	ssim_epoch = 0
	psnr_epoch = 0 
	for i in range(0, num_iterations): 
		 input_, target_, target_h_w_  = data_reader.get_next_batch_train(i) 
		 num_images += 1 
		 cost, _, lr, gl, predicted_images, output_h_w_before_, output_before_, output_h_w_, f_, after_ps_, features_h_w_ = sess.run([loss, opt, learning_rate, global_step, output, output_h_w_before, output_before, output_h_w, features, after_ps, features_h_w], feed_dict={input: input_ , target: target_, target_h_w: target_h_w_})   
		 # pdb.set_trace()
		 # for index in range(output_h_w_.shape[0]):
		 	# cv.imshow('hw', output_h_w_[index] / 255)
		 	# cv.imshow('output_h_w_before_', output_h_w_before_[index] / 255)
		 	# cv.imshow('outbefore', output_before_[index] / 255)
		 	# print(index)
		 	# cv.waitKey(0)
		 # for index in range(predicted_images.shape[0]):
		 	# cv.imshow('p', predicted_images[index] / 255)
		 	# cv.imshow('t', target_[index] / 255)
		 	# cv.imshow('b', output_before_[index] / 255)
		 	# cv.waitKey(0)
		 cv.imshow('i', input_[1] / 255)
		 cv.imshow('output_h_w_before_', output_h_w_before_[1] / 255)
		 cv.imshow('p', predicted_images[1] / 255)
		 cv.imshow('t', target_[1] / 255)
		 cv.waitKey(1000)
		 batch_loss += cost * data_reader.dim_depth 
		 ssim_batch, psnr_batch = utils.compute_ssim_psnr_batch(predicted_images, target_)
		 ssim_epoch += ssim_batch
		 psnr_epoch += psnr_batch
		 print("Epoch/Iteration/Global Iteration: {}/{}/{} ...".format(epoch, i, gl),"Training loss: {:.8f}".format(batch_loss/(num_images*data_reader.dim_depth)), "Learning rate:  {:.8f}".format(lr))  
		 
	merged_ = sess.run(merged, feed_dict={total_loss_placeholder: batch_loss/(num_images*data_reader.dim_depth), ssim_placeholder: ssim_epoch/num_images, psnr_placeholder: psnr_epoch/num_images, lr_placeholder : lr } )
	writer.add_summary(merged_, epoch)
	print('saving checkpoint...') 
    
	saver.save(sess, params.folder_data + params.ckpt_name + str(epoch))	

sess.close()