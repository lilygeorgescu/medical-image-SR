import tensorflow as tf
import numpy as np
import cv2 as cv 
import params
import utils
import pdb
 

params.show_params()

def predict(checkpoint_name, images=None, path_images=None, write_images=False):
    scale_factor = 2 #params.scale   
    
    if(type(images) != type(None)):
        original_image = images
    else:
        if(path_images == None):
            raise ValueError('if images is None path_images must not be none.') 
        original_image = utils.read_all_images_from_directory(path_images)   
     
    # standard resize
  
    # standard_resize = utils.resize_3d_image_standard(original_image, int(original_image.shape[0] * scale_factor), int(original_image.shape[1] * scale_factor), int(original_image.shape[2] * scale_factor), interpolation_method = params.interpolation_method)
    # to free momory
    # utils.write_3d_images(path_images, standard_resize, 'standard', scale_factor)
    # standard_resize= []
    
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input') 
    output_h_w, _ = params.network_architecture_H_W(input, params.kernel_size) 
    
    input_depth = tf.placeholder(tf.float32, (1, int(downscaled_image.shape[2] * params.scale), downscaled_image.shape[0], params.num_channels), name='input_depth')  
    output, _, _ = params.network_architecture_D(input_depth, params.kernel_size)   
    config = tf.ConfigProto(
            device_count = {'GPU': 0}
        ) 
    config.gpu_options.allow_growth = True
    intermediate_scale = 2
    num_iterations = np.log2(scale_factor, intermediate_scale)

    for i in range(num_iterations): 
        tf.reset_default_graph()
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

    if(write_images and path_images != None): 
        utils.write_3d_images(path_images, cnn_output, 'cnn', scale_factor)
        
            

# 'D:/disertatie/cnn/data/test/00001_0010/'
predict(checkpoint_name='./data_ckpt/model.ckpt299', path_images='D:/disertatie/results/var1-cnn-late-upscaling/output-images/00001_0010/4x/cnn/', write_images=True)