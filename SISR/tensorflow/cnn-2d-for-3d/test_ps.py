import tensorflow as tf
import numpy as np
import params
import pdb
import math


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


shape = (1, 64, 64, 4)
image = np.random.rand(1, 64, 64, 4)

r = 2

input = tf.placeholder(tf.float32, shape, name='input') 
output = PS(input, r)


sess = tf.Session()
res_tf = sess.run(output, feed_dict={input:image})

img_1 = PS_inference(image[0], r) 
print(np.sum(np.abs(np.round(img_1, 4) - np.round(res_tf[0], 4))))
assert np.sum(np.abs(np.round(img_1, 4) - np.round(res_tf[0], 4))) < 1e-3


# img_2 = PS_inference(image[1], r)
# assert np.sum(np.abs(np.round(img_2, 4) - np.round(res_tf[1], 4))) < 1e-3


