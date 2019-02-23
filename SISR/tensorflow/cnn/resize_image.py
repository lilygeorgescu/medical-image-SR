import tensorflow as tf
import numpy as np
import cv2 as cv
import networks as nets
import params

params.show_params()
scale_factor = params.scale 
image_name = params.image_name

original_image = cv.imread(image_name)
   
standard_resize = cv.resize(original_image, (0,0) ,fx = scale_factor, fy = scale_factor, interpolation = params.interpolation_method)

[H, W, C] = standard_resize.shape

input = tf.placeholder(tf.float32, (None, None, None, C), name='input') 
output, _ = nets.net_skip1(input, params.kernel_size)



config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('===========resuming from ' + tf.train.latest_checkpoint('./data/'))
saver.restore(sess,tf.train.latest_checkpoint('./data/'))

cnn_output = sess.run(output, feed_dict={input: [standard_resize]})
cnn_output = cnn_output[0,:,:,:]
 
cnn_output[np.where(cnn_output[:,:,:] > 255)] = 255
print(cnn_output.shape)
print(standard_resize.shape)
 
cv.imwrite("./output-images/test_standard_{}x{}".format(scale_factor, image_name),standard_resize)
cv.imwrite("./output-images/test_cnn_{}x{}".format(scale_factor, image_name),cnn_output)