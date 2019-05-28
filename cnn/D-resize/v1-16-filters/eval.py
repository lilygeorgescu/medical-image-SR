import tensorflow as tf
import numpy as np
import cv2 as cv 
import params
import utils
import pdb
import re
import os 

params.show_params()

config = tf.ConfigProto(
        device_count = {'GPU': 1}
    ) 

def run_network(downscaled_image, checkpoint):
    scale_factor = params.scale  
    
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    output = params.network_architecture(input, is_training=False) 
     

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint)
        saver.restore(sess, checkpoint)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output)   
        cnn_output = np.round(cnn_output) 
        cnn_output[cnn_output > 255] = 255 
        return cnn_output    
        
def predict(downscaled_image_1_2, downscaled_image_2_1, original_image, checkpoint): 

    scale_factor = params.scale    
    standard_resize = utils.resize_height_width_3d_image_standard(downscaled_image_1_2, int(downscaled_image_1_2.shape[1]), int(downscaled_image_1_2.shape[2])*scale_factor, interpolation_method = params.interpolation_method) 
    
    predict_1_2 = run_network(downscaled_image_1_2, checkpoint)
    tf.reset_default_graph()
    predict_2_1 = run_network(downscaled_image_2_1, checkpoint)
    
    predict_1_2 = np.transpose(predict_1_2, [2, 0, 1, 3]) 
    predict_2_1 = np.transpose(predict_2_1, [2, 1, 0, 3]) 
    standard_resize = np.transpose(standard_resize, [2, 0, 1, 3]) 
    cnn_output = 0.5 * (predict_1_2 + predict_2_1)
         
          
    ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(cnn_output, original_image)
    ssim_standard, psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image)

    return ssim_cnn, psnr_cnn, ssim_standard, psnr_standard 
   
        
def read_images(test_path):

    if use_original:
        add_to_path = 'original'
    else:
        add_to_path = 'transposed'
        
    test_images_gt = utils.read_all_directory_images_from_directory_test(test_path, add_to_path=add_to_path)
    test_images_1_2 = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='input_')    
    test_images_2_1 = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='input_2_1')  

    return test_images_gt, test_images_1_2, test_images_2_1
    
def compute_performance_indeces(test_path, test_images_gt, test_images_1_2, test_images_2_1, checkpoint, add_to_summary=True):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images_gt)): 
    
        if index == 2 and test_path.find('train') != -1:
            continue # an image has odd size
            
        ssim_cnn, psnr_cnn, ssim_standard, psnr_standard = predict(test_images_1_2[index], test_images_2_1[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images_gt[index].shape[0]
     
    print('standard {} --- psnr = {} ssim = {}'.format(test_path, psnr_standard_sum/num_images, ssim_standard_sum/num_images)) 
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))
    
    if test_path.find('test') != -1 and add_to_summary:
        tf.summary.scalar('psnr_standard', psnr_standard_sum/num_images) 
        tf.summary.scalar('psnr_cnn', psnr_cnn_sum/num_images)  
        tf.summary.scalar('ssim_standard', ssim_standard_sum/num_images)  
        tf.summary.scalar('ssim_cnn', ssim_cnn_sum/num_images)  
        merged = tf.summary.merge_all()
         
        writer = tf.summary.FileWriter('test.log') 
            
        epoch = re.findall(r'\d+', checkpoint)
        epoch = int(epoch[0])
        
        with tf.Session(config=config) as sess:
            merged_ = sess.run(merged)
            writer.add_summary(merged_, epoch)

use_original = True        
test_path = './../data/test' 
train_path = './data/train' 

test_images_gt, test_images_1_2, test_images_2_1 = read_images(test_path)
checkpoint = os.path.join(params.folder_data, 'model.ckpt%d' % 90) 

# checkpoint = tf.train.latest_checkpoint(params.folder_data)  
compute_performance_indeces(test_path, test_images_gt, test_images_1_2, test_images_2_1, checkpoint, add_to_summary=False)
exit()
# compute_performance_indeces(eval_path, eval_images, eval_images_gt, checkpoint)  


for i in range(65,  80):
    checkpoint = os.path.join(params.folder_data, 'model.ckpt%d' % i) 
    compute_performance_indeces(test_path, test_images_gt, test_images_1_2, test_images_2_1, checkpoint) 
 