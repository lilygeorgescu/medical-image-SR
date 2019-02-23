import tensorflow as tf
import numpy as np
import cv2 as cv
import random
import math
from sklearn.utils import shuffle
import pdb
import re

import networks as nets
import utils
import params

SHOW_IMAGES = False 
IS_RESTORE = False 

def batch_data(dim_patch, ground_truth, start, batch_size):
	end = start + batch_size 
	if( end > len(ground_truth)):
		end = len(ground_truth)
		batch_size = end - start
	
	input_images = np.zeros((batch_size, int(dim_patch / params.scale), int(dim_patch / params.scale), params.num_channels))
	output_images = np.zeros((batch_size, dim_patch, dim_patch, params.num_channels))
	
	for idx in range(start, end):
		image = ground_truth[idx]  
		im_H = image.shape[0]
		im_W = image.shape[1] 
		i = random.randint(0, im_H - dim_patch)
		j = random.randint(0, im_W - dim_patch) 
        
		if(params.num_channels == 1):
			input_images[idx - start, :, :, 0] = cv.resize(image[i:i + dim_patch, j:j + dim_patch], (0,0), fx = 1.0 / scale_factor, fy = 1.0 / scale_factor)
			output_images[idx - start, :, :, 0] = image[i:i + dim_patch, j:j + dim_patch]         
		else:
			input_images[idx - start, :, :, :] = cv.resize(image[i:i + dim_patch, j:j + dim_patch], (0,0), fx = 1.0 / scale_factor, fy = 1.0 / scale_factor)
			output_images[idx - start, :, :, :] = image[i:i + dim_patch, j:j + dim_patch] 
            
		if(SHOW_IMAGES):
			cv.imshow('input', input_images[idx - start, :, :, :]/255)
			cv.imshow('output', output_images[idx - start, :, :, :]/255)
			cv.waitKey(0)
		
	return input_images, output_images, batch_size

params.show_params() 
scale_factor = params.scale

# read all the images in a nd-array of size (num_images, height, width, channels)
image = utils.read_all_images_from_directory()  
images = []
  
min_scale = params.min_scale
max_scale = 1.0

ground_truth = []

# The augmentation is done by downscaling the test image I to many smaller versions of itself(I = I0,I1,I2,...,In)
for scale in np.arange(min_scale, max_scale + 0.01, 0.025):
    print('creating images with scale = {}'.format(scale))
    for idx_image in range(image.shape[0]):
        new_image = cv.resize(image[idx_image, :, :, :], (0,0), fx = scale, fy = scale)
        images.append(new_image)
	
# more augmentation
rotations = [90, 180, 270]
min_H = image.shape[1]
min_W = image.shape[2]

for im in images:	    
	 if(im.shape[0] < min_H):
	 	 min_H = im.shape[0]
	 if(im.shape[1] < min_W):
	 	 min_W = im.shape[1]
	  
	 ground_truth.append(im)
	 # flipped image  
	 ground_truth.append(cv.flip(im, 1))
	 # add rotation
	 for rot in rotations: 
	 	 rotetated_gt = utils.rotate(im, rot)
	 	 ground_truth.append(rotetated_gt) 
	 	 ground_truth.append(cv.flip(rotetated_gt, 1))
		
	 print('first image size = [{}] '.format(im.shape))
     
if(SHOW_IMAGES):
    for im in ground_truth:
        cv.imshow('im', im / 255 )
        cv.waitKey(1000) 
	
dim_patch = min(params.dim_patch, min_W, min_H)
	 
	 
# train 
batch_size = 32
dim_input = int(dim_patch / scale_factor)
input = tf.placeholder(tf.float32, (batch_size, dim_input, dim_input, params.num_channels), name='input')
target = tf.placeholder(tf.float32, (batch_size, dim_patch, dim_patch, params.num_channels), name='target')
output = nets.plain_net_late_upscaling(input, params.kernel_size) 
print('output shape is ', output.shape)
if(params.LOSS == params.L1_LOSS):
	loss = tf.reduce_mean(tf.abs(output - target)) 
if(params.LOSS == params.L2_LOSS):
	loss = tf.reduce_mean(tf.square(output - target)) 	
	
	
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = params.learning_rate

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.85, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

config = tf.ConfigProto(
        device_count = {'GPU': 1}
    ) 
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer()) 
saver = tf.train.Saver() 

total_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="total_loss")
lr_placeholder = tf.placeholder(tf.float32, shape=[], name="total_loss")
tf.summary.scalar('loss', total_loss_placeholder) 
tf.summary.scalar('learning_rate', lr_placeholder)  
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('train.log', sess.graph)
 

start = 0

if(IS_RESTORE):
    print('===========restoring from ' + tf.train.latest_checkpoint(params.folder_data))
    saver.restore(sess,tf.train.latest_checkpoint(params.folder_data))
    start = re.findall(r'\d+', tf.train.latest_checkpoint(params.folder_data))
    start = int(start[0]) + 1

num_images = len(ground_truth)
print('the number of images is ', num_images)
for e in range(start,params.num_epochs):
	batch_loss = 0  
	ground_truth = shuffle(ground_truth)
	random_indexes = np.random.permutation(len(ground_truth)) 
	num_iterations = math.floor(len(ground_truth) / batch_size)

	start = 0 
	for i in range(0,num_iterations): 
		 input_, target_, batch_size_ = batch_data(dim_patch, ground_truth, start, batch_size) 
		 print(input_.shape, target_.shape)
		 start = start + batch_size
		 cost, _, lr, gl = sess.run([loss, opt, learning_rate, global_step], feed_dict={input: input_ , target: target_})
		 batch_loss += cost * batch_size_ 
		 print("Epoch/Iteration/Global Iteration: {}/{}/{} ...".format(e, i, gl),"Training loss: {:.8f}".format(batch_loss/num_images), "Learning rate:  {:.8f}".format(lr)) 
 
		 
	merged_ = sess.run(merged, feed_dict={total_loss_placeholder: batch_loss/num_images, lr_placeholder : lr } )
	writer.add_summary(merged_, e)
	print('saving checkpoint...')
	saver.save(sess, params.folder_data + 'model.ckpt' + str(e))	
	


		
		
	
	

	 