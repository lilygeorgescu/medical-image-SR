import tensorflow as tf
import params
import numpy as np
import pdb
 
def plain_net_late_upscaling(im, kernel_size, num_layers = params.layers, is_inference = False):

	output = tf.contrib.layers.conv2d(im, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
	 
	for i in range(0, num_layers - 2):
		output = tf.contrib.layers.conv2d(output, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
        
	feature_map_for_ps = tf.contrib.layers.conv2d(output, num_outputs = params.num_channels * (params.scale ** 2), kernel_size = 3, stride = 1, padding='SAME', activation_fn=tf.nn.relu)  
	output_ = PS(feature_map_for_ps, params.scale)
	output_ = tf.contrib.layers.conv2d(output_, num_outputs = params.num_channels, kernel_size = 3, stride = 1, padding='SAME', activation_fn = None)
    
	return output_   
    
def PS_H_W(X, r): 
    X = _phase_shift(X, r)
    return X    
    
def _phase_shift(I, r):
    # Helper function with main phase shift operation 
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1 
 
    if(X.shape[0] == 1):
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r] 
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r 
        X = tf.reshape(X, [1, X.shape[0].value, X.shape[1].value, X.shape[2].value])
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r] 
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        X = tf.reshape(X, [1, X.shape[0].value, X.shape[1].value])            
    else:
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r] 
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r 
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r] 
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r 
             
    return tf.reshape(X, (bsize, a*r, b*r, 1))  
    
def SE_block(x, num_in, factor, name):
    with tf.name_scope(name) as scope:
        z = tf.contrib.layers.avg_pool2d(x, kernel_size = (np.int(x.shape[1]), np.int(x.shape[2])), stride = 1, padding='VALID')
        num_out = np.round(np.divide(num_in, factor))
        fc1 = tf.contrib.layers.conv2d(z, num_outputs = num_out, kernel_size = 1, stride = 1, padding='VALID', activation_fn = tf.nn.relu)
        fc2 = tf.contrib.layers.conv2d(fc1, num_outputs = num_in, kernel_size = 1, stride = 1, padding='VALID', activation_fn = tf.nn.sigmoid) 
        x_tilde = tf.multiply(x, fc2)
        return x_tilde	    


def custom_initializer(shape_list, dtype, partition_info): 
    # 0.9605
     
    return tf.ones(shape_list, dtype=dtype) * 0.000158
    
def SRCNN_late_upscaling_H_W(im, reuse=False): 
      
    output_1 = tf.layers.conv2d(im, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='conv2d')	  
    stride = 0
    # residual block
    output_2 = tf.layers.conv2d(output_1, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='conv2d_1') 
    output_3 = tf.layers.conv2d(output_2, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='conv2d_2')
    output_4 = tf.add(tf.multiply(output_1 , 1), output_3)
    
    
    output = tf.layers.conv2d(output_4, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='conv2d_3')
    output = tf.layers.conv2d(output, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='conv2d_4')     
    stride = 0 # [:, stride:-stride, stride:-stride, :]
    output = tf.add(tf.multiply(output_1, 1), output)
    
    feature_map_for_ps = tf.layers.conv2d(output, filters=params.num_channels * (params.scale ** 2), kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='conv2d_5')  
    # print(feature_map_for_ps.shape , output_1.shape)
    
    output_PS = PS_H_W(feature_map_for_ps, params.scale)  
    
    output_5 = tf.layers.conv2d(output_PS, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='last_layer_1', reuse=reuse)
     # residual block
    
    output_6 = tf.layers.conv2d(output_5, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='last_layer_2', reuse=reuse)
    output_7 = tf.layers.conv2d(output_6, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='last_layer_3', reuse=reuse)
    output_8 = tf.add(tf.multiply(output_5, 1), output_7, name='last_layer_4')
    
    output_9 = tf.layers.conv2d(output_8, filters=params.num_channels, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='last_layer_5', reuse=reuse)
      
    return output_PS, output_9  