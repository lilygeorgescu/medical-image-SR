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
 
# standard resize 
standard_resize = utils.resize_3d_image_standard(original_image, int(original_image.shape[0] * scale_factor), int(original_image.shape[1] * scale_factor), int(original_image.shape[2] * scale_factor), interpolation_method = params.interpolation_method)

# cnn resize 
# step 1 - resize the width and the height
image_resized_h_d = utils.resize_height_width_3d_image_standard(original_image, int(original_image.shape[1] * scale_factor), int(original_image.shape[2] * scale_factor), interpolation_method = params.interpolation_method) 

input = tf.placeholder(tf.float32, (None, None, None, params.num_channels), name='input') 
output, _ = nets.SE_net(input, params.kernel_size)



config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('===========restoring from ' + tf.train.latest_checkpoint(params.folder_data))
saver.restore(sess,tf.train.latest_checkpoint(params.folder_data))
 
# step 2 - apply cnn on each resized image, maybe as a batch => no memory :(
# cnn_output = sess.run(output, feed_dict={input: image_resized_h_d})
cnn_output = np.zeros((image_resized_h_d.shape))
num_images = cnn_output.shape[0]
for idx_img in range(num_images):
    res = sess.run(output, feed_dict={input: [image_resized_h_d[idx_img, :, :, :]]})
    cnn_output[idx_img, :, :, :] = res[0, :, :, :]

# step 3 - resize the depth with a standard method
cnn_output = utils.resize_depth_3d_image_standard(cnn_output, int(original_image.shape[0] * scale_factor), int(original_image.shape[1] * scale_factor), int(original_image.shape[2] * scale_factor), interpolation_method = params.interpolation_method) 
  
print(cnn_output.shape)
print(standard_resize.shape)
 
utils.write_3d_images(cnn_output, 'cnn')
utils.write_3d_images(standard_resize, 'standard')