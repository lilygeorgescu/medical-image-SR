import tensorflow as tf
import numpy as np
import cv2 as cv
import random
import math
from sklearn.utils import shuffle

import networks as nets
import utils
import params
 
SHOW_IMAGES = False

def batch_data(dim_patch, dim_depth, train_images, ground_truth, start, batch_size):
	end = start + batch_size 
	if( end > len(train_images)):
		end = len(train_images)
		batch_size = end - start
	
	input_images = np.zeros((batch_size, dim_depth, dim_patch, dim_patch, 1))
	output_images = np.zeros((batch_size, dim_depth, dim_patch, dim_patch, 1))
	
	for idx in range(start, end):
		image = train_images[idx] 
		gt_image = ground_truth[idx] 
		im_D = image.shape[0]
		im_H = image.shape[1]
		im_W = image.shape[2] 
		i = random.randint(0, im_H - dim_patch)
		j = random.randint(0, im_W - dim_patch) 
		k = random.randint(0, im_D - dim_depth) 
		input_images[idx - start, :, :, :, :] = image[k:k + dim_depth, i:i + dim_patch, j:j + dim_patch, :] 
		output_images[idx - start, :, :, :,:] = gt_image[k:k + dim_depth, i:i + dim_patch, j:j + dim_patch, :] 
	if(SHOW_IMAGES):
		for i in range(batch_size): 
			for j in range(dim_depth):
				cv.imshow('im_input', input_images[i, j, :, :, :] / 255)
				cv.imshow('im_output', output_images[i, j, :, :, :] / 255)
				cv.waitKey(0)
	
	return input_images, output_images, batch_size

params.show_params()
 
scale_factor = params.scale


image = utils.read_all_images_from_directory()  

# resize the original image 
image = utils.resize_3d_image_standard(image, int(image.shape[0] / scale_factor), int(image.shape[1] / scale_factor), int(image.shape[2] / scale_factor))

min_scale = params.min_scale
max_scale = 1.0

ground_truth = []
images = []

# The augmentation is done by downscaling the test image I to many smaller versions of itself(I = I0,I1,I2,...,In)
for scale in np.arange(min_scale, max_scale + 0.01, 0.025):
	print('creating image with scale = {}'.format(scale))
	new_image = utils.resize_3d_image_standard(image, max(1, int(image.shape[0] * scale)), int(image.shape[1] * scale), int(image.shape[2] * scale))
	images.append(new_image)
	
# upscale the image using a standard method	
train_images = []
rotations = [90, 180, 270]
min_D = image.shape[0]
min_H = image.shape[1]
min_W = image.shape[2]


for im in images:	  
	 downscaled_image = utils.resize_3d_image_standard(im, max(1, int(image.shape[0] / scale_factor)), int(image.shape[1] / scale_factor), int(image.shape[2] / scale_factor))
	 upscaled_image = utils.resize_3d_image_standard(downscaled_image, im.shape[0], im.shape[1], im.shape[2], params.interpolation_method)
     
	 if(upscaled_image.shape[1] < min_H):
	 	 min_H = upscaled_image.shape[1]
	 if(upscaled_image.shape[2] < min_W):
	 	 min_W = upscaled_image.shape[2]
	 if(upscaled_image.shape[0] < min_D):
	 	 min_D = upscaled_image.shape[0]
         
	 train_images.append(upscaled_image)
	 ground_truth.append(im)
	 # flipped image 
	 train_images.append(utils.flip_images(upscaled_image)) 
	 ground_truth.append(utils.flip_images(im))
     # add rotation
	 for rot in rotations:
	 	 rotated_image = utils.rotate_images(upscaled_image, rot)
	 	 train_images.append(rotated_image)
	 	 rotetated_gt = utils.rotate_images(im, rot)
	 	 ground_truth.append(rotetated_gt)
	 	 train_images.append(utils.flip_images(rotated_image))
	 	 ground_truth.append(utils.flip_images(rotetated_gt))
		
	 print('first image size = [{}] upscaled = [{}] downscaled = [{}]'.format( im.shape, upscaled_image.shape, downscaled_image.shape))

if(SHOW_IMAGES):
    for image in train_images:
        num_images = image.shape[0]
        for i in range(num_images):
            cv.imshow('im', image[i, :, :] / 255)
            cv.waitKey(0)
            
    for image in ground_truth:
        num_images = image.shape[0]
        for i in range(num_images):
            cv.imshow('im', image[i, :, :] / 255)
            cv.waitKey(0)
	
dim_patch = min(params.dim_patch, min_W, min_H)
dim_depth = min(params.dim_depth, min_D)	 
	 
# train      
input = tf.placeholder(tf.float32, (None, None, None, None, 1), name='input')
target = tf.placeholder(tf.float32, (None, None, None, None, 1), name='target')
output,out_ = nets.plain_net(input, params.kernel_size)

if(params.LOSS == params.L1_LOSS):
	loss = tf.reduce_mean(tf.abs(output - target)) 
if(params.LOSS == params.L2_LOSS):
	loss = tf.reduce_mean(tf.square(output - target)) 	
	
	
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = params.learning_rate

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.85, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
 
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
saver = tf.train.Saver() 

total_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="total_loss")
lr_placeholder = tf.placeholder(tf.float32, shape=[], name="total_loss")
tf.summary.scalar('batch_loss', total_loss_placeholder) 
tf.summary.scalar('learning_rate', lr_placeholder)  
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('train.log', sess.graph)
 
batch_size = 4

print(len(ground_truth))
for e in range(0,params.num_epochs):
	batch_loss = 0 
	iteration = 0
	train_images, ground_truth = shuffle(train_images, ground_truth)
	num_iterations = math.ceil(len(ground_truth) / batch_size)

	start = 0 
	for i in range(0,num_iterations):
	
		 input_, target_, batch_size_ = batch_data(dim_patch, dim_depth, train_images, ground_truth, start, batch_size) 
		 start = start + batch_size
		 cost, _, lr, gl, out  = sess.run([loss, opt, learning_rate, global_step,out_], feed_dict={input: input_ , target: target_})
		 batch_loss += cost * batch_size_ 
		 print("Epoch/Iteration/Global Iteration: {}/{}/{} ...".format(e,iteration,gl),"Training loss: {:.8f}".format(batch_loss),"Learning rate:  {:.8f}".format(lr)) 
		 iteration = iteration + 1
		 
	merged_ = sess.run(merged, feed_dict={total_loss_placeholder: batch_loss,lr_placeholder : lr } )
	writer.add_summary(merged_, e)
	print('saving checkpoint...')
	saver.save(sess, './data/model.ckpt' + str(e))	
	


		
		
	
	

	 