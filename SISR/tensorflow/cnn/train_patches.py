import tensorflow as tf
import numpy as np
import cv2 as cv
import random
import math
from sklearn.utils import shuffle

import networks as nets
import utils
import params



def batch_data(dim_patch, train_images, ground_truth, start, batch_size):
	end = start + batch_size 
	if( end > len(train_images)):
		end = len(train_images)
		batch_size = end - start
	
	input_images = np.zeros((batch_size, dim_patch, dim_patch, params.num_channels))
	output_images = np.zeros((batch_size, dim_patch, dim_patch, params.num_channels))
	
	for idx in range(start, end):
		image = train_images[idx]
		gt_image = ground_truth[idx] 
		im_H = image.shape[0]
		im_W = image.shape[1] 
		i = random.randint(0, im_H - dim_patch)
		j = random.randint(0, im_W - dim_patch) 
		if(params.num_channels == 1):
			input_images[idx - start, :, :, 0] = image[i:i + dim_patch,j:j + dim_patch] 
			output_images[idx - start, :, :, 0] = gt_image[i:i + dim_patch,j:j + dim_patch] 
		else:
			input_images[idx - start, :, :, :] = image[i:i + dim_patch,j:j + dim_patch] 
			output_images[idx - start, :, :, :] = gt_image[i:i + dim_patch,j:j + dim_patch] 
		# cv.imshow('a',input_images[idx - start, :, :, :]/255)
		# cv.imshow('b',output_images[idx - start, :, :, :]/255)
		# cv.waitKey(0)
		
	return input_images, output_images, batch_size

params.show_params()

image_name = params.image_name
scale_factor = params.scale

image = cv.imread(image_name)
images = []

if(params.num_channels == 1):
    image = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
else:
    image = cv.imread(image_name) 
# resize the original image
image = cv.resize(image, (0,0), fx = 1.0 / scale_factor, fy = 1.0 / scale_factor)


min_scale = params.min_scale
max_scale = 1.0

ground_truth = []

# The augmentation is done by downscaling the test image I to many smaller versions of itself(I = I0,I1,I2,...,In)
for scale in np.arange(min_scale, max_scale + 0.01, 0.025):
	print('creating image with scale = {}'.format(scale))
	new_image = cv.resize(image, (0,0), fx = scale, fy = scale)
	images.append(new_image)
	
# upscale the image using a standard method	
train_images = []
rotations = [90, 180, 270]
min_H = image.shape[0]
min_W = image.shape[1]

for im in images:	 
	 downscaled_image = cv.resize(im, (0,0), fx = 1.0 / scale_factor, fy = 1.0 / scale_factor)
	 upscaled_image = cv.resize(downscaled_image, (im.shape[1], im.shape[0]) , interpolation = params.interpolation_method)
	 
	 if(upscaled_image.shape[0] < min_H):
	 	 min_H = upscaled_image.shape[0]
	 if(upscaled_image.shape[1] < min_W):
	 	 min_W = upscaled_image.shape[1]
	 
	 train_images.append(upscaled_image)
	 ground_truth.append(im)
	 # flipped image 
	 train_images.append(cv.flip(upscaled_image, 1))
	 ground_truth.append(cv.flip(im, 1))
		 # add rotation
	 for rot in rotations:
	 	 rotated_image = utils.rotate(upscaled_image, rot)
	 	 train_images.append(rotated_image)
	 	 rotetated_gt = utils.rotate(im, rot)
	 	 ground_truth.append(rotetated_gt)
	 	 train_images.append(cv.flip(rotated_image, 1))
	 	 ground_truth.append(cv.flip(rotetated_gt, 1))
		
	 print('first image size = [{}] upscaled = [{}] downscaled = [{}]'.format( im.shape, upscaled_image.shape, downscaled_image.shape))

# for im in ground_truth:
	# cv.imshow('im',im)
	# cv.waitKey(1000) 
	
dim_patch = min(params.dim_patch, min_W, min_H)
	 
	 
# train 
input = tf.placeholder(tf.float32, (None, None, None, params.num_channels), name='input')
target = tf.placeholder(tf.float32, (None, None, None, params.num_channels), name='target')
output, out_ = nets.net_skip1(input, params.kernel_size)

if(params.LOSS == params.L1_LOSS):
	loss = tf.reduce_mean(tf.abs(output - target)) 
if(params.LOSS == params.L2_LOSS):
	loss = tf.reduce_mean(tf.square(output - target)) 	
	
	
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = params.learning_rate

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           50, 0.8, staircase=True)
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
 
batch_size = 32

print(len(ground_truth))
for e in range(0,params.num_epochs):
	batch_loss = 0 
	iteration = 0
	train_images, ground_truth = shuffle(train_images, ground_truth)
	random_indexes = np.random.permutation(len(train_images)) 
	num_iterations = math.ceil(len(ground_truth) / batch_size)

	start = 0 
	for i in range(0,num_iterations):
	
		 input_, target_, batch_size_ = batch_data(dim_patch, train_images, ground_truth, start, batch_size) 
		 start = start + batch_size
		 cost, _, lr, gl, out  = sess.run([loss, opt, learning_rate, global_step,out_], feed_dict={input: input_ , target: target_})
		 batch_loss += cost * batch_size_ 
		 print("Epoch/Iteration/Global Iteration: {}/{}/{} ...".format(e,iteration,gl),"Training loss: {:.8f}".format(batch_loss),"Learning rate:  {:.8f}".format(lr)) 
		 iteration = iteration + 1
		 
	merged_ = sess.run(merged, feed_dict={total_loss_placeholder: batch_loss,lr_placeholder : lr } )
	writer.add_summary(merged_, e)
	print('saving checkpoint...')
	saver.save(sess, './data/model.ckpt' + str(e))	
	


		
		
	
	

	 