import tensorflow as tf
import params 

def plain_net(im, kernel_size, num_layers = params.layers):

	output = tf.contrib.layers.conv3d(im, num_outputs = 64, kernel_size = kernel_size, stride=1, padding='SAME', activation_fn = tf.nn.relu)
	 
	for i in range(0, num_layers - 2):
		output = tf.contrib.layers.conv3d(output, num_outputs = 64, kernel_size = kernel_size, stride=1, padding='SAME', activation_fn = tf.nn.relu)
	
	output_ = tf.contrib.layers.conv3d(output, num_outputs = 1, kernel_size = kernel_size, stride=1, padding='SAME', activation_fn = None)
	output = output_ + im
	 
	return output, output_
 


