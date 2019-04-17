import tensorflow as tf
import numpy as np
import cv2 as cv 
import params
import utils
import pdb
 

params.show_params()

def predict(downscaled_image=None, original_image=None, path_images=None, path_original_images=None, write_images=False):
    scale_factor = params.scale   
    
    if path_images is None and downscaled_image is None:
            raise ValueError('if images is None path_images must not be none.') 
    if path_original_images is None and original_image is None:
            raise ValueError('if path_original_images is None original_image must not be none.')
            
    original_image = utils.read_all_images_from_directory(path_original_images)
    mean = np.loadtxt('mean.txt')
    # standard resize
    downscaled_image = utils.read_all_images_from_directory(path_images) # utils.resize_height_width_3d_image_standard(original_image, int(original_image.shape[1] / scale_factor), int(original_image.shape[2] / scale_factor))
    standard_resize = utils.resize_height_width_3d_image_standard(downscaled_image, int(original_image.shape[1]), int(original_image.shape[2]), interpolation_method = params.interpolation_method)
    downscaled_image = downscaled_image / 255  
    downscaled_image = downscaled_image - mean
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    output = params.network_architecture(input, params.kernel_size) 
     
    config = tf.ConfigProto(
            device_count = {'GPU': 1}
        ) 
    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('===========restoring from ' + tf.train.latest_checkpoint(params.folder_data))
        saver.restore(sess, tf.train.latest_checkpoint(params.folder_data))
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output)  
        print(cnn_output.shape)
        print(standard_resize.shape)
        cnn_output = np.round(cnn_output * 255) 
        ## eval
        if(params.tf_version >= 1.10):
            original = tf.placeholder(tf.float32, (None, None, None, 1), name='input') 
            reconstructed = tf.placeholder(tf.float32, (None, None, None, 1), name='input') 

            psnr = tf.image.psnr(original, reconstructed, max_val = 255)
            ssim = tf.image.ssim_multiscale(original, reconstructed, max_val = 255)
          
        num_images = cnn_output.shape[0]
        sum_ssim_standard = 0
        sum_ssim_cnn = 0
        sum_psnr_standard = 0
        sum_psnr_cnn = 0
        stride = 11
        for index in range(num_images): 
           if(params.tf_version >= 1.10):
                [psnr_standard, ssim_standard] = sess.run([psnr, ssim], feed_dict={original: [original_image[index, :, :]], reconstructed: [standard_resize[index, :, :]]})
                [psnr_cnn, ssim_cnn] = sess.run([psnr,ssim], feed_dict={original: [original_image[index, :, :]], reconstructed: [cnn_output[index, :, :]]})
           else:
                psnr_standard = utils.psnr(original_image[index, :, :], standard_resize[index, :, :])
                ssim_standard = utils.ssim(original_image[index, :, :], standard_resize[index, :, :]) 
                psnr_cnn = utils.psnr(original_image[index, stride:-stride, stride:-stride], cnn_output[index, :, :])
                ssim_cnn = utils.ssim(original_image[index, stride:-stride, stride:-stride], cnn_output[index, :, :]) 
                
           sum_ssim_cnn += ssim_cnn
           sum_psnr_cnn += psnr_cnn
           sum_ssim_standard += ssim_standard
           sum_psnr_standard += psnr_standard
           
        print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard, sum_ssim_standard)) 
        print('cnn --- psnr = {} ssim = {}'.format(sum_psnr_cnn, sum_ssim_cnn)) 
        print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard / num_images, sum_ssim_standard / num_images)) 
        print('cnn --- psnr = {} ssim = {}'.format(sum_psnr_cnn / num_images, sum_ssim_cnn / num_images))  
        
        if(write_images and path_images != None):
            utils.write_3d_images(path_images, cnn_output, 'cnn')
            utils.write_3d_images(path_images, standard_resize, 'standard')
            

predict(path_images='./data/train_/00001_0002/input_', path_original_images='./data/train_/00001_0002/', write_images=True)