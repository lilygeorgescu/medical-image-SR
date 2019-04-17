  
import os
import io 
import tensorflow as tf

from PIL import Image 

 
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
def create_tf_example(image_path_gt):
    with tf.gfile.GFile(image_path_gt, 'rb') as fid:
        encoded_jpeg_gt = fid.read()  
    width_gt, height_gt = Image.open(io.BytesIO(encoded_jpeg_gt)).size 
  
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image_gt/height': int64_feature(height_gt),
        'image_gt/width': int64_feature(width_gt), 
        'image_gt/encoded': bytes_feature(encoded_jpeg_gt) 
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter('train.record')
    path_train_images = 'images'
    images = os.listdir(path_train_images)
    
    for image in images: 
        gt_path = os.path.join(path_train_images, image)  
        print(gt_path)
        tf_example = create_tf_example(gt_path)
        writer.write(tf_example.SerializeToString())

    writer.close() 
    print('Successfully created the TFRecords')


if __name__ == '__main__':
    tf.app.run()