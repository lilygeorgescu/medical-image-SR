import tensorflow as tf
import numpy as np
import cv2 as cv
import networks as nets
import utils
import params

params.show_params()

image_name = params.image_name
scale_factor = params.scale

image = cv.imread(image_name)
images = []


# resize the original image
image = cv.resize(image, (0,0), fx = 1.0 / scale_factor, fy = 1.0 / scale_factor)


min_scale = 0.8
max_scale = 1.0

ground_truth = []

# The augmentation is done by downscaling the test image I to many smaller versions of itself(I = I0,I1,I2,...,In)
for scale in np.arange(min_scale, max_scale + 0.05, 0.1):
	print('creating image with scale = {}'.format(scale))
	new_image = cv.resize(image, (0,0), fx = scale, fy = scale)
	images.append(new_image)
	
# upscale the image using a standard method	
train_images = []
rotations = [90, 180, 270]
for im in images:	 
	 downscaled_image = cv.resize(im, (0,0), fx = 1.0 / scale_factor, fy = 1.0 / scale_factor)
	 upscaled_image = cv.resize(downscaled_image, (im.shape[1], im.shape[0]) , interpolation = params.interpolation_method)
	 train_images.append(upscaled_image)
	 ground_truth.append(im)
	 # flipped image 
	 train_images.append(cv.flip(upscaled_image, 1))
	 ground_truth.append(cv.flip(im, 1))
		 # add rotation
	 for rot in rotations:
	 	 rotated_image = utils.rotate(upscaled_image,rot)
	 	 train_images.append(rotated_image)
	 	 rotetated_gt = utils.rotate(im,rot)
	 	 ground_truth.append(rotetated_gt)
	 	 train_images.append(cv.flip(rotated_image, 1))
	 	 ground_truth.append(cv.flip(rotetated_gt, 1))
		
	 print('first image size = [{}] upscaled = [{}] downscaled = [{}]'.format( im.shape, upscaled_image.shape, downscaled_image.shape))

# for im in ground_truth:
	# cv.imshow('im',im)
	# cv.waitKey(1000) 
	

	 
	 
# train 
input = tf.placeholder(tf.float32, (None, None, None, 3), name='input')
target = tf.placeholder(tf.float32, (None, None, None, 3), name='target')
output = nets.net_skip1(input, params.kernel_size)

if(params.LOSS == params.L1_LOSS):
	loss = tf.reduce_mean(tf.abs(output - target)) 
if(params.LOSS == params.L2_LOSS):
	loss = tf.reduce_mean(tf.square(output - target)) 	
	
	
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = params.learning_rate

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           2000, 0.9, staircase=True)
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


for e in range(0,params.num_epochs):
	batch_loss = 0 
	iteration = 0
	random_indexes = np.random.permutation(len(train_images)) 
	
	
	for i in range(0,len(train_images)):
		 cost, _, lr, gl  = sess.run([loss, opt,learning_rate, global_step], feed_dict={input: [train_images[random_indexes[i]]]  ,
						target: [ground_truth[random_indexes[i]]] })
		 batch_loss += cost
		 
		 print("Epoch/Iteration/Global Iteration: {}/{}/{} ...".format(e,iteration,gl),"Training loss: {:.8f}".format(batch_loss),"Learning rate:  {:.8f}".format(lr)) 
		 iteration = iteration + 1
	merged_ = sess.run(merged, feed_dict={total_loss_placeholder: batch_loss,lr_placeholder : lr } )
	writer.add_summary(merged_, e)
	print('saving checkpoint...')
	saver.save(sess, './data/model.ckpt' + str(e))	
	
	 

	 