import cv2 as cv
import utils
import pdb
import numpy as np
import tensorflow as tf

images_path = './data/train/00001_0002/'
images = utils.read_all_images_from_directory(images_path)
im = np.transpose(images, [1, 2, 0, 3])
im_patch = im[250]
cv.imshow('a', im_patch)
cv.waitKey()
# cv.imwrite('a.png', im_patch)
# pdb.set_trace()

# check training procedure
def _phase_shift_D(I, r):
    # Helper function with main phase shift operation 
    bsize, a, b, c = I.shape #I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, 1, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1  
    if(X.shape[0] == 1): 
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r] 
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r 
        X = tf.reshape(X, [1, X.shape[0].value, X.shape[1].value, 1])
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r] 
        X = tf.concat([x for x in X], 1)  # bsize, a*r, b*r 
    else:
        
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r] 
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r 
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r] 
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r 
              
    return tf.reshape(X, (bsize, a*1, b*r, 1))  

res = _phase_shift_D(im, 2)
with tf.Session() as sess:
    res_ = sess.run(res)
    pdb.set_trace()