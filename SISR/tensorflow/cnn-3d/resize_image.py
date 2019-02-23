import tensorflow as tf
import numpy as np
import cv2 as cv
import networks as nets
import params
import utils

params.show_params()
scale_factor = params.scale  

utils.create_folders()
original_image = utils.read_all_images_from_directory()     

standard_resize = utils.resize_3d_image_standard(original_image, int(original_image.shape[0] * scale_factor), int(original_image.shape[1] * scale_factor), int(original_image.shape[2] * scale_factor))
 

input = tf.placeholder(tf.float32, (None, None, None, None, 1), name='input') 
output, _ = nets.plain_net(input, params.kernel_size)



config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('===========resuming from ' + tf.train.latest_checkpoint('./data/'))
saver.restore(sess,tf.train.latest_checkpoint('./data/'))

cnn_output = sess.run(output, feed_dict={input: [standard_resize]})
cnn_output = cnn_output[0, :, :, :, :]
  
print(cnn_output.shape)
print(standard_resize.shape)
 
utils.write_3d_images(cnn_output, 'cnn')
utils.write_3d_images(standard_resize, 'standard')