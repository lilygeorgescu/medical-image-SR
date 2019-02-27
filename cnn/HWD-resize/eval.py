import tensorflow as tf
import numpy as np
import cv2 as cv
import random
import math
from sklearn.utils import shuffle
import pdb
import re
import data_reader as reader
import time
import os


import networks as nets
import utils
import params

def run_network(images, ground_truth, checkpoint_name):
    '''
        images, ground_truth are nd-arrays of size [batch_size(num_images=depth), heigth, width, num_channels]
    '''  
    
    tf.reset_default_graph() 
    input = tf.placeholder(tf.float32, (1, images.shape[1], images.shape[2], params.num_channels), name='input') 
    output_h_w, _ = params.network_architecture_H_W(input, params.kernel_size) 
    
    input_depth = tf.placeholder(tf.float32, (1, ground_truth.shape[2], images.shape[0], params.num_channels), name='input_depth')  
    output, _, _ = params.network_architecture_D(input_depth, params.kernel_size)   
    
    predicted = tf.placeholder(tf.float32, (ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2], params.num_channels), name='predicted')
    target = tf.placeholder(tf.float32, (ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2], params.num_channels), name='target')
    # loss computed based on the original 3d image and the 3d image  
    if(params.LOSS == params.L1_LOSS):
        loss = tf.reduce_mean(tf.abs(predicted - target)) 
    if(params.LOSS == params.L2_LOSS):
        loss = tf.reduce_mean(tf.square(predicted - target))
        
    # restore values
    saver = tf.train.Saver()
    
    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    ) 
    with tf.Session(config=config) as sess:    
        saver.restore(sess, checkpoint_name)
        # resize on height and witdh
        output_h_w_ = np.zeros((images.shape[0], ground_truth.shape[1], ground_truth.shape[2], ground_truth.shape[3]))
        for i in range(images.shape[0]):
            output_h_w_[i] = sess.run(output_h_w, feed_dict={input: [images[i]]})    
            
        # resize on depth    
        output_h_w_ = np.transpose(output_h_w_, [1, 2, 0, 3])    
        output_h_w_d = np.zeros((ground_truth.shape[1], ground_truth.shape[2], ground_truth.shape[0], params.num_channels))
        for i in range(images.shape[0]):
            output_h_w_d[i] = sess.run(output, feed_dict={input_depth: [output_h_w_[i]]})  
            
        output_3d_rezised = np.transpose(output_h_w_d, [2, 0, 1, 3])   
        
        cost = sess.run(loss, feed_dict={predicted: output_3d_rezised , target: ground_truth})    
        
        ssim_batch, psnr_batch = utils.compute_ssim_psnr_batch(output_3d_rezised, ground_truth) 
          
        return cost, ssim_batch, psnr_batch
    
  
  
def eval(data_reader, checkpoint_name=tf.train.latest_checkpoint(params.folder_data)): 
     
    num_images = 0
    cost = 0
    ssim = 0
    psnr = 0 
    epoch = int(re.findall(r'\d+', checkpoint_name)[0])
    
    # for every nd-array in the list
    for i in range(data_reader.num_eval_images):  
        images = data_reader.eval_images[i]
        ground_truth = data_reader.eval_images_gt[i]
        num_images += ground_truth.shape[0]
        loss, ssim_batch, psnr_batch = run_network(images, ground_truth, checkpoint_name)
        cost += loss * ground_truth.shape[0]
        ssim += ssim_batch
        psnr += psnr_batch   
    print(ssim, psnr, num_images)       

    tf.summary.scalar('loss', cost/num_images)  
    tf.summary.scalar('ssim', ssim/num_images)  
    tf.summary.scalar('psnr', psnr/num_images)  
    merged = tf.summary.merge_all()
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    ) 
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('eval.log')
        merged_ = sess.run(merged)
        writer.add_summary(merged_, epoch)    
        
    print('eval---epoch: {} loss: {} ssim: {} psnr: {} '.format(epoch, cost/num_images, ssim/num_images, psnr/num_images))
     
    
def test(data_reader, checkpoint_name=tf.train.latest_checkpoint(params.folder_data)): 
     
    num_images = 0
    cost = 0
    ssim = 0
    psnr = 0 
    epoch = int(re.findall(r'\d+', checkpoint_name)[0])
    
    # for every nd-array in the list
    for i in range(data_reader.num_test_images):  
        images = data_reader.test_images[i]
        ground_truth = data_reader.test_images_gt[i] 
        num_images += ground_truth.shape[0]
        loss, ssim_batch, psnr_batch = run_network(images, ground_truth, checkpoint_name)
        cost += loss * ground_truth.shape[0]
        ssim += ssim_batch
        psnr += psnr_batch
    print(ssim_batch, psnr_batch, num_images)   
    tf.summary.scalar('loss', cost/num_images)  
    tf.summary.scalar('ssim', ssim/num_images)  
    tf.summary.scalar('psnr', psnr/num_images)  
    merged = tf.summary.merge_all()
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    ) 
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('test.log')
        merged_ = sess.run(merged)
        writer.add_summary(merged_, epoch)    
        
    print('test---epoch: {} loss: {} ssim: {} psnr: {} '.format(epoch, cost/num_images, ssim/num_images, psnr/num_images))
     


def run_eval_test(data_reader):
    while(True):
        latest_checkpoint = tf.train.latest_checkpoint(params.folder_data)
        if(latest_checkpoint == None):
            print('sleeping for 60 sec')
            time.sleep(60)
            continue
        # check if it was already tested
        if(os.path.isfile(params.latest_ckpt_filename)):
            latest_checkpoint_tested = np.loadtxt(params.latest_ckpt_filename, dtype="str")
            if(latest_checkpoint_tested == latest_checkpoint):
                print('sleeping for 60 sec')
                time.sleep(60)
            else:
                eval(data_reader, latest_checkpoint)
                test(data_reader, latest_checkpoint)
                np.savetxt(params.latest_ckpt_filename, [latest_checkpoint], delimiter=" ", fmt="%s")
        else:
               # eval(data_reader, latest_checkpoint)
                test(data_reader, latest_checkpoint)
                np.savetxt(params.latest_ckpt_filename, [latest_checkpoint], delimiter=" ", fmt="%s")            
                
    

    
data_reader = reader.DataReader('./data/train', './data/validation', './data/test', is_training=False)
eval(data_reader, checkpoint_name='./data_ckpt/model.ckpt49')
run_eval_test(data_reader)    