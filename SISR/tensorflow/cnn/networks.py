import tensorflow as tf
import params

def plain_net(im):
	conv1 = tf.contrib.layers.conv2d(im, num_outputs = 3, kernel_size = 3, stride=1, padding='SAME',activation_fn=tf.nn.relu)
	
	conv2 = tf.contrib.layers.conv2d(conv1, num_outputs = 3, kernel_size = 3, stride=1, padding='SAME',activation_fn=tf.nn.relu)
	
	conv3 = tf.contrib.layers.conv2d(conv2, num_outputs = 3, kernel_size = 3, stride=1, padding='SAME',activation_fn=tf.nn.relu)
	
	return conv3;

def net_skip1(im, kernel_size):

	output = tf.contrib.layers.conv2d(im, num_outputs = 64, kernel_size = kernel_size, stride=1, padding='SAME',activation_fn=tf.nn.relu)
	 
	for i in range(0,params.layers - 2):
		output = tf.contrib.layers.conv2d(output, num_outputs = 64, kernel_size = kernel_size, stride=1, padding='SAME',activation_fn=tf.nn.relu)
	
	output_ = tf.contrib.layers.conv2d(output, num_outputs = params.num_channels, kernel_size = kernel_size, stride=1, padding='SAME',activation_fn = None)
	output = output_ + im
	 
	return output, output_;

def  net_skip2(im):
	conv1 = tf.contrib.layers.conv2d(im, num_outputs = 128, kernel_size = 3, stride=1, padding='SAME',activation_fn=tf.nn.relu)
	
	conv2 = tf.contrib.layers.conv2d(conv1, num_outputs = 128, kernel_size = 3, stride=1, padding='SAME',activation_fn=tf.nn.relu)
	
	skip2 = conv2 + conv1
	
	conv3 = tf.contrib.layers.conv2d(skip2, num_outputs = 3, kernel_size = 3, stride=1, padding='SAME',activation_fn=tf.nn.relu)
	
	output = conv3 + im
	
	return output;


def _phase_shift(I, r):
    # Helper function with main phase shift operation 
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1 
    if(params.tf_version >= 1.10):
        
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
    else: 
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

def PS(X, r, color=False):
  # Main OP that you can arbitrarily use in you tensorflow code
  color = (params.num_channels == 3)
  if color:
   if(params.tf_version >= 1.10):
        Xc = tf.split(3, 3, X)
        X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
   else:
        Xc = tf.split(X ,3, 3 )
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
  else:
    X = _phase_shift(X, r)
  return X
  
def plain_net_late_upscaling(im, kernel_size, num_layers = params.layers, is_inference = False):

	output = tf.contrib.layers.conv2d(im, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
	 
	for i in range(0, num_layers - 2):
		output = tf.contrib.layers.conv2d(output, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
        
	feature_map_for_ps = tf.contrib.layers.conv2d(output, num_outputs = params.num_channels * (params.scale ** 2), kernel_size = 3, stride = 1, padding='SAME', activation_fn=tf.nn.relu)  
	output_ = PS(feature_map_for_ps, params.scale)
	output_ = tf.contrib.layers.conv2d(output_, num_outputs = params.num_channels, kernel_size = 3, stride = 1, padding='SAME', activation_fn = None)
    
	return output_   
  
def SRCNN_late_upscaling(im, kernel_size, num_layers = params.layers): 
	reg = 0.005
	output = tf.layers.conv2d(im, filters = 16, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
	  
	output = tf.layers.conv2d(output, filters = 16, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
        
	feature_map_for_ps = tf.layers.conv2d(output, filters = params.num_channels * (params.scale ** 2), kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))  
	output_ = PS(feature_map_for_ps, params.scale)
	output_ = tf.layers.conv2d(output_, filters = params.num_channels, kernel_size = 3, strides = 1, padding='SAME', activation = None, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
    
	return output_ 