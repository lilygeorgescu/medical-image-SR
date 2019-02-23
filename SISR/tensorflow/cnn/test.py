import tensorflow as tf
import numpy as np
import cv2 as cv
import networks as nets
import params
import utils


params.show_params()
scale_factor = params.scale 
image_name = params.image_name
 
if(params.num_channels == 1):
    original_image = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
else:
    original_image = cv.imread(image_name)   
downscaled_image = cv.resize(original_image, (0,0), fx = 1.0 / scale_factor, fy = 1.0 / scale_factor)
standard_resize = cv.resize(downscaled_image, (original_image.shape[1], original_image.shape[0]) , interpolation = params.interpolation_method)
 

input = tf.placeholder(tf.float32, (None, None, None, params.num_channels), name='input') 
output, _ = nets.net_skip1(input, params.kernel_size)

if(params.num_channels == 1):
    standard_resize = np.expand_dims(standard_resize, 3) 

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('===========resuming from ' + tf.train.latest_checkpoint('./data/'))
saver.restore(sess,tf.train.latest_checkpoint('./data/'))

cnn_output = sess.run(output, feed_dict={input: [standard_resize]})
cnn_output = cnn_output[0,:,:,:]
 
cnn_output[np.where(cnn_output[:,:,:] > 255)] = 255
cnn_output[np.where(cnn_output[:,:,:] < 0)] = 0
print(cnn_output.shape)
print(standard_resize.shape)


## eval
if(params.tf_version >= 1.10):
    original = tf.placeholder(tf.float32, (None, None, None, params.num_channels), name='input') 
    reconstructed = tf.placeholder(tf.float32, (None, None, None, params.num_channels), name='input') 

    psnr = tf.image.psnr(original, reconstructed, max_val = 255)
    ssim = tf.image.ssim_multiscale(original, reconstructed, max_val = 255)

    [psnr_standard, ssim_standard] = sess.run([psnr, ssim], feed_dict={original: [original_image], reconstructed: [standard_resize]})
    [psnr_cnn, ssim_cnn] = sess.run([psnr, ssim], feed_dict={original: [original_image], reconstructed: [cnn_output]})
else:
    psnr_cnn = utils.psnr(original_image, cnn_output)
    ssim_cnn = utils.ssim(original_image, cnn_output)
    psnr_standard = utils.psnr(original_image, standard_resize)
    ssim_standard = utils.ssim(original_image, standard_resize)

    
print('standard --- psnr = {} ssim = {}'.format(psnr_standard, ssim_standard)) 
print('cnn --- psnr = {} ssim = {}'.format(psnr_cnn, ssim_cnn)) 
 
cv.imwrite("./output-images/standard_{}x{}".format(scale_factor, image_name),standard_resize)
cv.imwrite("./output-images/cnn_{}x{}".format(scale_factor, image_name),cnn_output)