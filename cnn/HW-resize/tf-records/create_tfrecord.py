  
import os
import io 
import tensorflow as tf

from PIL import Image 

 
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
def create_tf_example(image_path_gt, image_path_in):
    with tf.gfile.GFile(image_path_gt, 'rb') as fid:
        encoded_jpeg_gt = fid.read()  
    width_gt, height_gt = Image.open(io.BytesIO(encoded_jpeg_gt)).size 
 
    with tf.gfile.GFile(image_path_in, 'rb') as fid:
        encoded_jpeg_in = fid.read()  
    width_in, height_in = Image.open(io.BytesIO(encoded_jpeg_in)).size 
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image_gt/height': int64_feature(height_gt),
        'image_gt/width': int64_feature(width_gt), 
        'image_gt/encoded': bytes_feature(encoded_jpeg_gt),
        'image_in/height': int64_feature(height_in),
        'image_in/width': int64_feature(width_in), 
        'image_in/encoded': bytes_feature(encoded_jpeg_in)
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter('train.record')
    path_train_images = 'D:\\disertatie\\code\\cnn\\HW-resize\\data\\train'
    folders = os.listdir(path_train_images)
    
    for folder in folders:
        num_images_folder = len(os.listdir(os.path.join(path_train_images, folder, 'gt')))
        for id_image in range(1, num_images_folder + 1): 
            in_path = os.path.join(path_train_images, folder, 'input', '%d.png' % id_image)
            gt_path = os.path.join(path_train_images, folder, 'gt', '%d.png' % id_image)
            print(in_path)
            print(gt_path)
            tf_example = create_tf_example(gt_path, in_path)
            writer.write(tf_example.SerializeToString())

    writer.close() 
    print('Successfully created the TFRecords')


if __name__ == '__main__':
    tf.app.run()