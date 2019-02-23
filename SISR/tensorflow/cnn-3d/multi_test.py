import tensorflow as tf
import numpy as np
import cv2 as cv
import networks as nets
import params
import utils

params.show_params()
scale_factor = params.scale 
image_name = params.image_name

original_image = cv.imread(image_name)
  
downscaled_image = cv.resize(original_image, (0,0), fx = 1.0 / scale_factor, fy = 1.0 / scale_factor)
standard_resize = cv.resize(downscaled_image, (original_image.shape[1], original_image.shape[0]) , interpolation = params.interpolation_method)

[H, W, C] = standard_resize.shape

input = tf.placeholder(tf.float32, (None, None, None, C), name='input') 
output, _ = nets.net_skip1(input, params.kernel_size)



config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('===========resuming from ' + tf.train.latest_checkpoint('./data/'))
saver.restore(sess,tf.train.latest_checkpoint('./data/'))

rotations = [90, 180, 270]

test_images = []
test_images.append(standard_resize)

for rot in rotations:
	rotated_image = utils.rotate(standard_resize, rot)
	test_images.append(rotated_image)
	
#flip
test_images.append(cv.flip(standard_resize, 1))

for rot in rotations:
	rotated_image = utils.rotate(standard_resize, rot)
	test_images.append(cv.flip(rotated_image, 1))
	
	
# for im in test_images:
	# cv.imshow('a',im/255)
	# cv.waitKey(0)
	
output_cnn_images = []

for image in test_images:
	cnn_output = sess.run(output, feed_dict={input: [image]})
	cnn_output = cnn_output[0,:,:,:] 
	cnn_output[np.where(cnn_output[:,:,:] > 255)] = 255
	output_cnn_images.append(cnn_output) 
	
	
# back to the original position

images = []
images.append(output_cnn_images[0])
index = 1
for rot in rotations:
	rotated_image = utils.rotate(output_cnn_images[index], -rot)
	index = index + 1
	images.append(rotated_image)
	
images.append(cv.flip(output_cnn_images[4], 1))

index = 5
for rot in rotations:
	flipped_image = cv.flip(output_cnn_images[index],  1)
	rotated_image = utils.rotate(flipped_image, 360 - rot)
	index = index + 1
	images.append(rotated_image)
	
cnn_output = np.zeros((original_image.shape[0], original_image.shape[1], 3))

num_images = len(images)
for idx in range(0,num_images): 
	# cv.imshow('a',im/255)
	cnn_output = cnn_output + images[idx]
	print(images[idx].shape)
	# cv.waitKey(0)
	
cnn_output = cnn_output / len(images)
print(cnn_output.shape)

## eval
original = tf.placeholder(tf.float32, (None, None, None, C), name='input') 
reconstructed = tf.placeholder(tf.float32, (None, None, None, C), name='input') 

psnr = tf.image.psnr(original, reconstructed, max_val = 255)
ssim = tf.image.ssim_multiscale(original, reconstructed, max_val = 255)

[psnr_standard, ssim_standard] = sess.run([psnr,ssim], feed_dict={original: [original_image], reconstructed: [standard_resize]})
[psnr_cnn, ssim_cnn] = sess.run([psnr,ssim], feed_dict={original: [original_image], reconstructed: [cnn_output]})

print('standard --- psnr = {} ssim = {}'.format(psnr_standard, ssim_standard)) 
print('cnn --- psnr = {} ssim = {}'.format(psnr_cnn, ssim_cnn)) 
 
cv.imwrite("./output-images/standard_{}x{}".format(scale_factor, image_name),standard_resize)
cv.imwrite("./output-images/cnn_{}x{}".format(scale_factor, image_name),cnn_output)
	
	