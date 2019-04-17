import tensorflow as tf
import pdb
from skimage.data import imread
import io
import timeit
import cv2 as cv
import os

tfrecords_filename = 'train.record'
# opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

reader = tf.TFRecordReader(tfrecords_filename)
filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
keys, values = reader.read_up_to(filename_queue, 2)
pdb.set_trace()
# record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename) 
# start_time = timeit.default_timer()
# for string_record in record_iterator:
    # example = tf.train.Example()
    # example.ParseFromString(string_record)
    # height = int(example.features.feature['image_gt/height']
                                 # .int64_list
                                 # .value[0])
    
    # width = int(example.features.feature['image_gt/width']
                                # .int64_list
                                # .value[0])
    
    # img_string = (example.features.feature['image_gt/encoded'].bytes_list.value[0])
    # im = imread(io.BytesIO(img_string)) 
 
        
# end_time = timeit.default_timer()   
# print('total time ', end_time - start_time)

 
 
# normal read    
path_train_images = 'images'
start_time = timeit.default_timer() 
images = os.listdir(path_train_images)

for image in images: 
    gt_path = os.path.join(path_train_images, image)  
    image = cv.imread(gt_path) 
    w = image.shape[:2]
    
end_time = timeit.default_timer()   
print('total time ', end_time - start_time)    
    
# _, serialized_example = reader.read(filename_queue)
# feature_set = {'image_gt/encoded': tf.FixedLenFeature([], tf.string),
               # 'image_gt/height': tf.FixedLenFeature([], tf.int64),
               # 'image_gt/width': tf.FixedLenFeature([], tf.int64),
               # 'image_in/encoded': tf.FixedLenFeature([], tf.string),
               # 'image_in/height': tf.FixedLenFeature([], tf.int64),
               # 'image_in/width': tf.FixedLenFeature([], tf.int64),
           # } 
           
# features = tf.parse_single_example( serialized_example, features= feature_set )
# label = features['image_gt/height']
# image = features['image_gt/encoded']
# print('before session')
# with tf.Session() as sess:
  # print('after session')
  # print(sess.run([label]))