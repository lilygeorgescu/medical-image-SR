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

# def run_network(images, ground_truth, checkpoint_name):
    # '''
        # images, ground_truth are nd-arrays of size [batch_size(num_images=depth), heigth, width, num_channels]
    # '''  
    
    # tf.reset_default_graph()
    
    # placeholders, input image and ground truth
    # input = tf.placeholder(tf.float32, (1, images.shape[1], images.shape[2], params.num_channels), name='input')
    # target = tf.placeholder(tf.float32, (ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2], params.num_channels), name='target')
    # output = params.network_architecture(input, params.kernel_size)  
    
    # placeholder for the 3D reconstructed image
    # output_3d_rezised_ph = tf.placeholder(tf.float32, (ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2], params.num_channels), name='target')
    
    # loss computed based on the original 3d image and the 3d image computed by resizing the image on height and width with nn and on depth with a standard method
    # if(params.LOSS == params.L1_LOSS):
        # loss = tf.reduce_mean(tf.abs(output_3d_rezised_ph - target)) 
    # if(params.LOSS == params.L2_LOSS):
        # loss = tf.reduce_mean(tf.square(output_3d_rezised_ph - target))
    # restore values
    # saver = tf.train.Saver()
    
    # config = tf.ConfigProto(
        # device_count = {'GPU': 1}
    # ) 
    # stride = 10
    # with tf.Session(config=config) as sess:    
        # saver.restore(sess, checkpoint_name)
        # resize on height and witdh
        # output_ = np.zeros((ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2], params.num_channels))
        # for i in range(images.shape[0]):
            # output_[i] = sess.run(output, feed_dict={input: [images[i]]})   
        # resize the image on depth with a standard method
        # output_3d_rezised = utils.resize_depth_3d_image_standard(output_, ground_truth.shape[0], ground_truth.shape[1], ground_truth.shape[2], interpolation_method = params.interpolation_method)
        # cost = sess.run(loss, feed_dict={output_3d_rezised_ph: output_3d_rezised, target: ground_truth[:, stride:-stride, stride:-stride, :]})
        
        # ssim_batch, psnr_batch = utils.compute_ssim_psnr_batch(output_3d_rezised, ground_truth[:, stride:-stride, stride:-stride, :]) 
          
        # return cost, ssim_batch, psnr_batch
    
def run_network(images, ground_truth, checkpoint_name):
    '''
        images, ground_truth are nd-arrays of size [batch_size(num_images=depth), heigth, width, num_channels]
    '''  
    
    tf.reset_default_graph()
    stride = 10
    # placeholders, input image and ground truth 
    input = tf.placeholder(tf.float32, (1, images.shape[1], images.shape[2], params.num_channels), name='input')
    target = tf.placeholder(tf.float32, (1, ground_truth.shape[1] - 2*stride, ground_truth.shape[2] - 2*stride, params.num_channels), name='target')
    output = params.network_architecture(input, params.kernel_size)  
      
    # loss computed based on the original 3d image and the 3d image computed by resizing the image on height and width with nn and on depth with a standard method
    if(params.LOSS == params.L1_LOSS):
        loss = tf.reduce_mean(tf.abs(output - target)) 
    if(params.LOSS == params.L2_LOSS):
        loss = tf.reduce_mean(tf.square(output - target))
    # restore values
    saver = tf.train.Saver()
    
    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    ) 
    
    with tf.Session(config=config) as sess:    
        saver.restore(sess, checkpoint_name) 
        all_cost = 0
        ssim_batch = 0
        psnr_batch = 0
        num_images = len(images)
        for idx_img in range(num_images):
            gt_image = ground_truth[idx_img][stride:-stride, stride:-stride, :]
            output_, cost = sess.run([output, loss], feed_dict={input:[images[idx_img]], target: [gt_image]})      
            all_cost += cost 
            ssim, psnr = utils.compute_ssim_psnr_batch(output_, [gt_image]) 
            ssim_batch += ssim
            psnr_batch += psnr
            
        return cost, ssim_batch, psnr_batch  
  
def eval(data_reader, checkpoint_name=tf.train.latest_checkpoint(params.folder_data)): 
     
    num_images = len(data_reader.eval_images)
    epoch = int(re.findall(r'\d+', checkpoint_name)[0])
 
    cost, ssim, psnr = run_network(data_reader.eval_images, data_reader.eval_images_gt, checkpoint_name)
       
    print(ssim, psnr, psnr)       

    tf.summary.scalar('loss', cost/num_images)  
    tf.summary.scalar('ssim', ssim/num_images)  
    tf.summary.scalar('psnr', psnr/num_images)  
    merged = tf.summary.merge_all()
    config = tf.ConfigProto( device_count = {'GPU': 1}  ) 
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('eval.log')
        merged_ = sess.run(merged)
        writer.add_summary(merged_, epoch)     
        
    print('eval---epoch: {} loss: {} ssim: {} psnr: {} '.format(epoch, cost/num_images, ssim/num_images, psnr/num_images))
     
    
def test(data_reader, checkpoint_name=tf.train.latest_checkpoint(params.folder_data)): 
     
    num_images = len(data_reader.test_images)
    epoch = int(re.findall(r'\d+', checkpoint_name)[0]) 
    
    cost, ssim, psnr = run_network(data_reader.test_images, data_reader.test_images_gt, checkpoint_name)
       
    print(ssim, psnr, psnr)       
    tf.summary.scalar('loss', cost/num_images)  
    tf.summary.scalar('ssim', ssim/num_images)  
    tf.summary.scalar('psnr', psnr/num_images)  
    merged = tf.summary.merge_all()
    config = tf.ConfigProto(
        device_count = {'GPU': 1}
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
                eval(data_reader, latest_checkpoint)
                test(data_reader, latest_checkpoint)
                np.savetxt(params.latest_ckpt_filename, [latest_checkpoint], delimiter=" ", fmt="%s")            
                
    

    
data_reader = reader.DataReader('./data/train', './data/validation', './data/test', is_training=False)
# eval(data_reader, checkpoint_name='./data_ckpt/model.ckpt49')
run_eval_test(data_reader)    