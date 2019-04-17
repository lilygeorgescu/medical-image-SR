import tensorflow as tf
import params
import pdb
 
def _phase_shift_D(I, r):
    # Helper function with main phase shift operation 
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, 1, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1    
    if(X.shape[0] == 1):   
            X = tf.split(X, a, 1)  # a, [bsize, b, r, r] 
            X = tf.concat([tf.expand_dims(tf.squeeze(x), 0) for x in X], 1)  # bsize, b, a*r, r 
            X = tf.reshape(X, [X.shape[0].value, X.shape[1].value, X.shape[2].value, 1])            
            X = tf.split(X, b, 1)  # b, [bsize, a*r, r] 
            X = tf.concat([tf.expand_dims(tf.squeeze(x), 0) for x in X], 1)
    else: 
            X = tf.split(X, a, 1)  # a, [bsize, b, r, r] 
            X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r 
            X = tf.reshape(X, [X.shape[0].value, X.shape[1].value, X.shape[2].value, 1])            
            X = tf.split(X, b, 1)  # b, [bsize, a*r, r] 
            X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r 
             
        
    return tf.reshape(X, (bsize, a*1, b*r, 1))    

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

def PS_H_W(X, r): 
    X = _phase_shift(X, r)
    return X
  
def PS_D(X, r): 
    X = _phase_shift_D(X, r)
    return X 
  
def SRCNN_late_upscaling_H_W(im, kernel_size, num_layers = params.layers): 
    reg = 0.005 
    # first layer  
    
    output = tf.layers.conv2d(im, filters = 32, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))	
    
    output = tf.layers.conv2d(output, filters = 64, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))	     
    
    output = tf.layers.conv2d(output, filters = 64, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))	  
    
    output = tf.layers.conv2d(output, filters = 32, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))  
    
    feature_map_for_ps = tf.layers.conv2d(output, filters = params.num_channels * (params.scale ** 2), kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))  
    
    output_ = PS_H_W(feature_map_for_ps, params.scale)
    
    output_ = tf.layers.conv2d(output_, filters = params.num_channels, kernel_size = 3, strides = 1, padding='SAME', activation = None, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
    
    return output_, feature_map_for_ps
    
def SRCNN_late_upscaling_D(im, kernel_size, num_layers = params.layers): 
    reg = 0.005 
    # first layer  
    
    output = tf.layers.conv2d(im, filters = 16, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
    
    output = tf.layers.conv2d(output, filters = 32, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))	   
    
    output = tf.layers.conv2d(output, filters = 32, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))	 
    
    output = tf.layers.conv2d(output, filters = 16, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
  
    feature_map_for_ps = tf.layers.conv2d(output, filters = params.num_channels * params.scale, kernel_size = 1, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))  
    
    output__ = PS_D(feature_map_for_ps, params.scale)
    
    output_ = tf.layers.conv2d(output__, filters = params.num_channels, kernel_size = 3, strides = 1, padding='SAME', activation=tf.nn.relu, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=reg)) 
    
    return output_