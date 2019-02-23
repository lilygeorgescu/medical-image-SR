import tensorflow as tf
import numpy as np
import cv2 as cv
import networks as nets
import params
import utils
import conv3d_operations

params.show_params()
scale_factor = params.scale  

utils.create_folders()
original_image = utils.read_all_images_from_directory()    

downscaled_image = utils.resize_3d_image_standard(original_image, int(original_image.shape[0] / scale_factor), int(original_image.shape[1] / scale_factor), int(original_image.shape[2] / scale_factor))

standard_resize = utils.resize_3d_image_standard(downscaled_image, int(original_image.shape[0]), int(original_image.shape[1]), int(original_image.shape[2]))
 

cnn_output = conv3d_operations.conv(standard_resize)
cnn_output = cnn_output[0, :, :, :, :]
  
print(cnn_output.shape)
print(standard_resize.shape)
 
## eval
config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
sess = tf.Session(config=config)

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
    [psnr_standard, ssim_standard] = sess.run([psnr, ssim], feed_dict={original: [original_image[index, :, :]], reconstructed: [standard_resize[index, :, :]]})
    sum_ssim_standard += ssim_standard
    sum_psnr_standard += psnr_standard
    [psnr_cnn, ssim_cnn] = sess.run([psnr,ssim], feed_dict={original: [original_image[index, :, :]], reconstructed: [cnn_output[index, :, :]]})
    sum_ssim_cnn += ssim_cnn
    sum_psnr_cnn += psnr_cnn
    
print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard, sum_ssim_standard)) 
print('cnn --- psnr = {} ssim = {}'.format(sum_psnr_cnn, sum_ssim_cnn)) 
print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard / num_images, sum_ssim_standard / num_images)) 
print('cnn --- psnr = {} ssim = {}'.format(sum_psnr_cnn / num_images, sum_ssim_cnn / num_images))  
utils.write_3d_images(cnn_output, 'cnn')
utils.write_3d_images(standard_resize, 'standard')