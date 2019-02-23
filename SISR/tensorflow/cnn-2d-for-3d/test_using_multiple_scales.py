import tensorflow as tf
import numpy as np
import cv2 as cv
import networks as nets
import params
import utils
import pdb

 
tf_version = 1.02
params.show_params()
scale_factor = 4 # params.scale  

utils.create_folders()
original_image = utils.read_all_images_from_directory()   
 
# standard resize
downscaled_image = utils.resize_3d_image_standard(original_image, int(original_image.shape[0] / scale_factor), int(original_image.shape[1] / scale_factor), int(original_image.shape[2] / scale_factor))
standard_resize = utils.resize_3d_image_standard(downscaled_image, int(original_image.shape[0]), int(original_image.shape[1]), int(original_image.shape[2]), interpolation_method = params.interpolation_method)

# session configuration 
input = tf.placeholder(tf.float32, (None, None, None, params.num_channels), name='input') 
output, _ = nets.plain_net(input, params.kernel_size)

config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('===========restoring from ' + tf.train.latest_checkpoint(params.folder_data))
saver.restore(sess,tf.train.latest_checkpoint(params.folder_data))

intermediate_scale = 2
num_iterations = int(scale_factor / intermediate_scale)

for i in range(num_iterations): 
    # cnn resize 
    # step 1 - resize the width and the height 
    if(i == intermediate_scale - 1): 
        image_resized_h_d = utils.resize_height_width_3d_image_standard(downscaled_image, int(original_image.shape[1]), int(original_image.shape[2]),interpolation_method = params.interpolation_method)  
    else:
        image_resized_h_d = utils.resize_height_width_3d_image_standard(downscaled_image, int(downscaled_image.shape[1] * intermediate_scale), int(downscaled_image.shape[2] * intermediate_scale), interpolation_method = params.interpolation_method)  
        
    # step 2 - apply cnn on each resized image, maybe as a batch
    cnn_output = sess.run(output, feed_dict={input: image_resized_h_d})

    # step 3 - resize the depth with a standard method
    if(i == intermediate_scale - 1):
        cnn_output = utils.resize_depth_3d_image_standard(cnn_output, int(original_image.shape[0]), int(original_image.shape[1]), int(original_image.shape[2]), interpolation_method = params.interpolation_method) 
    else:
        cnn_output = utils.resize_depth_3d_image_standard(cnn_output, int(downscaled_image.shape[0] * intermediate_scale), int(downscaled_image.shape[1] * intermediate_scale), int(downscaled_image.shape[2] * intermediate_scale), interpolation_method = params.interpolation_method) 
    downscaled_image = cnn_output 
    
print(cnn_output.shape)
print(standard_resize.shape)
 
## eval
if(tf_version >= 1.10):
    original = tf.placeholder(tf.float32, (None, None, None, 1), name='input') 
    reconstructed = tf.placeholder(tf.float32, (None, None, None, 1), name='input') 

    psnr = tf.image.psnr(original, reconstructed, max_val = 255)
    ssim = tf.image.ssim_multiscale(original, reconstructed, max_val = 255)

num_images = cnn_output.shape[0]
sum_ssim_standard = 0
sum_ssim_cnn = 0
sum_psnr_standard = 0
sum_psnr_cnn = 0

for index in range(num_images):
    if(tf_version >= 1.10):
        [psnr_standard, ssim_standard] = sess.run([psnr, ssim], feed_dict={original: [original_image[index, :, :]], reconstructed: [standard_resize[index, :, :]]})
        [psnr_cnn, ssim_cnn] = sess.run([psnr,ssim], feed_dict={original: [original_image[index, :, :]], reconstructed: [cnn_output[index, :, :]]})
    else:
        psnr_standard = utils.psnr(original_image[index, :, :], standard_resize[index, :, :])
        ssim_standard = utils.ssim(original_image[index, :, :], standard_resize[index, :, :])
        psnr_cnn = utils.psnr(original_image[index, :, :], cnn_output[index, :, :])
        ssim_cnn = utils.ssim(original_image[index, :, :], cnn_output[index, :, :])
        
    
    sum_ssim_standard += ssim_standard
    sum_psnr_standard += psnr_standard
     
    sum_ssim_cnn += ssim_cnn
    sum_psnr_cnn += psnr_cnn
    
print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard, sum_ssim_standard)) 
print('cnn --- psnr = {} ssim = {}'.format(sum_psnr_cnn, sum_ssim_cnn)) 
print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard / num_images, sum_ssim_standard / num_images)) 
print('cnn --- psnr = {} ssim = {}'.format(sum_psnr_cnn / num_images, sum_ssim_cnn / num_images))  
utils.write_3d_images(cnn_output, 'cnn')
utils.write_3d_images(standard_resize, 'standard')