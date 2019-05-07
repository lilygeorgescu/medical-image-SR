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
    downscaled_image = downscaled_image
    downscaled_image = downscaled_image - mean
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    output = params.network_architecture(input) 
     
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
        cnn_output = np.round(cnn_output) 
 
 
        stride = None
        
        ssim_cnn, psnr_cnn = utils.compute_ssim_psnr(cnn_output, original_image, stride=stride)
        ssim_standard, psnr_standard = utils.compute_ssim_psnr(standard_resize, original_image, stride=stride)
        
        print('standard --- psnr = {} ssim = {}'.format(psnr_standard, ssim_standard)) 
        print('cnn --- psnr = {} ssim = {}'.format(psnr_cnn, ssim_cnn))  
        
        if(write_images and path_images != None):
            utils.write_3d_images(path_images, cnn_output, 'cnn')
            utils.write_3d_images(path_images, standard_resize, 'standard')
            

predict(path_images='./data/test/00001_0009/input', path_original_images='./data/test/00001_0009/', write_images=True)
tf.reset_default_graph()
predict(path_images='./data/test/00001_0010/input', path_original_images='./data/test/00001_0010/', write_images=False)
tf.reset_default_graph()
predict(path_images='./data/test/00001_0011/input', path_original_images='./data/test/00001_0011/', write_images=False)
tf.reset_default_graph()
predict(path_images='./data/train/00001_0003/input_', path_original_images='./data/train/00001_0003/', write_images=True)  