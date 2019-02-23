import tensorflow as tf
import numpy as np
import params 
import pdb
 

def SE_block(x, num_in, factor, name):
	with tf.name_scope(name) as scope: 
		# z = tf.contrib.layers.avg_pool2d(x, kernel_size = np.int(x.shape[1]), stride = 1, padding='VALID') 
        # cannot use avg pooling because the width and height are not defined
		z = tf.reduce_mean(x, axis=[1, 2])
        # now, we need to reshape z because its shape is (?, num_in)
		z = tf.reshape(z, [-1, 1, 1, num_in]) 
		num_out = np.round(np.divide(num_in, factor))
		fc1 = tf.contrib.layers.conv2d(z, num_outputs = num_out, kernel_size = 1, stride = 1, padding='VALID', activation_fn=tf.nn.relu)
		fc2 = tf.contrib.layers.conv2d(fc1, num_outputs = num_in, kernel_size = 1, stride = 1, padding='VALID', activation_fn=tf.nn.sigmoid)
		x_tilde = tf.multiply(x, fc2)
		return x_tilde	
        
def plain_net(im, kernel_size, num_layers = params.layers):

	output = tf.contrib.layers.conv2d(im, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
	 
	for i in range(0, num_layers - 2):
		output = tf.contrib.layers.conv2d(output, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
	
	output_ = tf.contrib.layers.conv2d(output, num_outputs = params.num_channels, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn = None)
	output = output_ + im
	 
	return output, output_;
    
    
def SE_net(im, kernel_size, num_layers = params.layers):

	output = tf.contrib.layers.conv2d(im, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
	output = SE_block(output, 64, 4, 'SE') 
	for i in range(0, num_layers - 2):
		output = tf.contrib.layers.conv2d(output, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
		output = SE_block(output, 64, 4, 'SE_{}'.format(i)) 
	output_ = tf.contrib.layers.conv2d(output, num_outputs = params.num_channels, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn = None)
	output = output_ + im
	 
	return output, output_;
    
def add_new_dim(X):
    if(len(X.shape) == 3):
        X = tf.reshape(X, [1, int(X.shape[0]), int(X.shape[1]), int(X.shape[2])])
    return X
    
def PS_inference(I, r): 
  assert r>0
  r = int(r)
  O = np.zeros((int(I.shape[0]*r), int(I.shape[1]*r), int(I.shape[2]/(r*2))))
  for x in range(O.shape[0]):
    for y in range(O.shape[1]):
      for c in range(O.shape[2]):
        c += 1
        a = int(math.floor(x/r))
        b = int(math.floor(y/r))
        d = int(c*r*(y%r) + c*(x%r)) 
        O[x, y, c-1] = I[a, b, d]
  return O    
    
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
  if color:
    Xc = tf.split(3, 3, X)
    X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
  else:
    X = _phase_shift(X, r)
  return X

    
def RIR_SE_late_upscale_net(im, kernel_size, dim_patch_H, dim_patch_W = None, num_layers = params.layers):

	if(dim_patch_W == None):
		dim_patch_W = dim_patch_H
	layers = []
	with tf.name_scope('block_1') as scope:     
		layers.append(tf.contrib.layers.conv2d(im, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu))
		layers.append(SE_block(layers[-1], 64, 4, 'SE'))
    
	for i in range(0, num_layers - 2):
		with tf.name_scope('block_%d' % (i + 1)) as scope:  
			layers.append(tf.contrib.layers.conv2d(layers[-1], num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu))
			layers.append(SE_block(layers[-1], 64, 4, 'SE_{}'.format(i))) 
        # add shorter skip connections every 4 layers
		# with tf.name_scope('skip_%d' % (i + 1)) as scope:  
			# if(len(layers) % 8 == 0):
				# layers.append(layers[-1] + layers[-8])
           
        
	with tf.name_scope('upscale') as scope:         
		layers.append(tf.contrib.layers.conv2d(layers[-1], num_outputs = params.num_channels, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn = None))
        # add the input image
		small_image = layers[-1] + im
        # late upscaling, the size is dim_patch * scale 
		upscaled_image = tf.image.resize_images(small_image, size = (int(dim_patch_H), int(dim_patch_W)), method = tf.image.ResizeMethod.BICUBIC) 
		layers.append(upscaled_image)
		# apply another conv layer
		output = tf.contrib.layers.conv2d(layers[-1], num_outputs = params.num_channels, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=None)
		print(layers)
        
		return output, layers[-1];
 
def plain_net_late_upscaling(im, kernel_size, num_layers = params.layers, is_inference = False):

	output = tf.contrib.layers.conv2d(im, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
	 
	for i in range(0, num_layers - 3):
		output = tf.contrib.layers.conv2d(output, num_outputs = 64, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)
        
	feature_map_for_ps = tf.contrib.layers.conv2d(output, num_outputs = params.num_channels * (params.scale ** 2), kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn=tf.nn.relu)  
	output_ = PS(feature_map_for_ps, params.scale)
	output_ = tf.contrib.layers.conv2d(output_, num_outputs = params.num_channels, kernel_size = kernel_size, stride = 1, padding='SAME', activation_fn = None)
    
	return output_ 
