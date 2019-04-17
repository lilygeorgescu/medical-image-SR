import tensorflow as tf
import numpy as np
import cv2 as cv 
import params
import utils
import pdb
 

params.show_params()

def predict(images=None, path_images=None, write_images=False, compare_with_ground_truth=True):
    scale_factor = params.scale   
    
    if(images is not None):
        original_image = images
    else:
        if(path_images is None):
            raise ValueError('if images is None path_images must not be none.') 
        original_image = utils.read_all_images_from_directory(path_images)   
        
    # standard resize
    if(compare_with_ground_truth):
        downscaled_image = utils.resize_3d_image_standard(original_image, int(original_image.shape[0] / scale_factor), int(original_image.shape[1] / scale_factor), int(original_image.shape[2] / scale_factor))
        standard_resize = utils.resize_3d_image_standard(downscaled_image, int(original_image.shape[0]), int(original_image.shape[1]), int(original_image.shape[2]), interpolation_method = params.interpolation_method)
    else:
        downscaled_image = original_image
        
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input') 
    output_h_w = params.network_architecture_H_W(input, params.kernel_size) 
    
    input_depth = tf.placeholder(tf.float32, (1, int(downscaled_image.shape[2] * params.scale), downscaled_image.shape[0], params.num_channels), name='input_depth')  
    output = params.network_architecture_D(input_depth, params.kernel_size)   
    
    predicted = tf.placeholder(tf.float32, (original_image.shape[0], original_image.shape[1], original_image.shape[2], params.num_channels), name='predicted') 
    # loss computed based on the original 3d image and the 3d image  
    if(params.LOSS == params.L1_LOSS):
        loss = tf.reduce_mean(tf.abs(predicted - original_image)) 
    if(params.LOSS == params.L2_LOSS):
        loss = tf.reduce_mean(tf.square(predicted - original_image))
        
    # restore values
    saver = tf.train.Saver()
    
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    ) 
    with tf.Session(config=config) as sess:   
        print('restoring {}...'.format(tf.train.latest_checkpoint(params.folder_data)))
        saver.restore(sess, tf.train.latest_checkpoint(params.folder_data))
        print('restored {}...'.format(tf.train.latest_checkpoint(params.folder_data)))
        # resize on height and witdh
        output_h_w_ = np.zeros((downscaled_image.shape[0], downscaled_image.shape[1]*params.scale, downscaled_image.shape[2]*params.scale, params.num_channels))
        for i in range(downscaled_image.shape[0]):
            output_h_w_[i] = sess.run(output_h_w, feed_dict={input: [downscaled_image[i]]})[0]     
        # resize on depth 
        output_h_w_ = np.transpose(output_h_w_, [1, 2, 0, 3])    
        output_h_w_d = np.zeros((downscaled_image.shape[1]*params.scale, downscaled_image.shape[2]*params.scale, downscaled_image.shape[0]*params.scale, params.num_channels)) 
        for i in range(output_h_w_.shape[0]): 
            output_h_w_d[i] = sess.run(output, feed_dict={input_depth: [output_h_w_[i]]})[0]   
            
        output_3d_resized = np.transpose(output_h_w_d, [2, 0, 1, 3])    
        
        if(compare_with_ground_truth):
            cost = sess.run(loss, feed_dict={predicted: output_3d_resized})     
            
            sum_ssim_cnn, sum_psnr_cnn = utils.compute_ssim_psnr_batch(output_3d_resized, original_image) 
            sum_ssim_standard, sum_psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image) 
            num_images = output_3d_resized.shape[0]
            print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard, sum_ssim_standard)) 
            print('cnn --- psnr = {} ssim = {}'.format(sum_psnr_cnn, sum_ssim_cnn)) 
            print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard / num_images, sum_ssim_standard / num_images)) 
            print('cnn --- psnr = {} ssim = {}'.format(sum_psnr_cnn / num_images, sum_ssim_cnn / num_images))  
        
        if(write_images and path_images != None):
            utils.write_3d_images(path_images, output_3d_resized, 'cnn')
        if(write_images and path_images != None and compare_with_ground_truth):
            utils.write_3d_images(path_images, standard_resize, 'standard')
            

predict(path_images='./data/train/00001_0002/', write_images=True, compare_with_ground_truth=True)