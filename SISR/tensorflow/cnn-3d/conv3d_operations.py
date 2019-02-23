import numpy as np
import tensorflow as tf
import utils
import networks as nets
import params
import pdb

def conv(image, num_layers = params.layers):          

    # (batch_size, depth, height, width, filters)  
    input = tf.placeholder(tf.float32, (None, None, None, None, None), name='input')
    # (batch_size, depth, height, width, filters) 
    filter = tf.placeholder(tf.float32, (None, None, None, None, None), name='target')

    output = tf.nn.conv3d(input, filter, strides=[1, 1, 1, 1, 1], padding='SAME')
    output_relu = tf.nn.relu(output)
     
    # image = utils.read_all_images_from_directory()
    # scale = params.scale
    # image = utils.resize_3d_image_standard(image, max(1, int(image.shape[0] * scale)), int(image.shape[1] * scale), int(image.shape[2] * scale))
    
    # I dont need it, but I must use it to restore the weights
    input_net = tf.placeholder(tf.float32, (None, None, None, None, 1), name='input_net')
    _, _ = nets.plain_net(input_net, params.kernel_size)
      
    config = tf.ConfigProto(
            device_count = {'GPU': 1}
        )
        
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer()) 
    saver = tf.train.Saver() 
    print('===========restoring from ' + tf.train.latest_checkpoint('./data/'))
    saver.restore(sess,tf.train.latest_checkpoint('./data/'))
    features = [image] 
    for layer in range(num_layers):
        print('layer is {}'.format(layer))
        if(layer == 0):
            # (batch_size, depth, height, width, filters) 
            w = tf.get_default_graph().get_tensor_by_name('Conv/weights:0')
            b = tf.get_default_graph().get_tensor_by_name('Conv/biases:0') 
            num_filters = w.shape[-1]
            # (batch_size, depth, height, width, filters) 
            new_features = np.zeros((1, image.shape[0], image.shape[1], image.shape[2], w.shape[-1]))
        else:
            # (batch_size, depth, height, width, filters) 
            w = tf.get_default_graph().get_tensor_by_name('Conv_{}/weights:0'.format(layer))
            b = tf.get_default_graph().get_tensor_by_name('Conv_{}/biases:0'.format(layer)) 
            num_filters = w.shape[-1]
            features = new_features 
            # (batch_size, depth, height, width, filters) 
            new_features = np.zeros((1, image.shape[0], image.shape[1], image.shape[2], w.shape[-1])) 
            
        print('the number of filters is {}'.format(num_filters))
        for index_filter in range(num_filters): 
            # (batch_size, depth, height, width, filters) 
            w_ = np.zeros((w.shape[0], w.shape[1], w.shape[2], w.shape[3], 1)) 
            # (batch_size, depth, height, width, filters) 
            w_[:, :, :, :, 0] = (w.eval(session = sess))[:, :, :, :, index_filter]  
            
            if(layer == num_layers - 1): # if it is the last layer dont apply relu
                result = sess.run(output, feed_dict = {input : features, filter : w_}) 
            else:    
                result = sess.run(output_relu, feed_dict = {input : features, filter : w_}) 
                
            result = result  + (b.eval(session = sess))[index_filter] 
            # (batch_size, depth, height, width, filters) 
            new_features[0, :, :, :, index_filter] = result[0, :, :, :, 0] 
            
    new_features = new_features + np.array([image])        
    
    return new_features 
    
# res = conv() 
# utils.write_3d_images(res[0, :, :, :, :], 'test_cnn')