from utils import *
import params
import tensorflow as tf
import pdb
import cv2 as cv 

def predict(downscaled_image, original_image, checkpoint):
 
    # network for original image
    config = tf.ConfigProto(
            device_count = {'GPU': 0}
        ) 
    
    sess_or = tf.Session(config=config)
    with sess_or:
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        output_or = params.network_architecture(input_or) 
        
    sess_tr = tf.Session(config=config)
    with sess_tr:
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        output_tr = params.network_architecture(input_tr, reuse=True)     
     
    saver = tf.train.Saver()
    print('restoring from ' + checkpoint)
    saver.restore(sess_or, checkpoint)
    saver.restore(sess_tr, checkpoint)
    
    num_images = downscaled_image.shape[0]
    cnn_output = []
    for image in downscaled_image:
        out_images = [] 
        
        # original 0 
        res = sess_or.run(output_or, {input_or: [image]})[0]
        out_images.append(res)
        
        # flip 0 
        res = sess_or.run(output_or, {input_or: [flip_image(image)]})[0]
        out_images.append(reverse_flip_image(res))
         
        # original 180
        rot180_image = rotate_image_180(image)
        res = sess_or.run(output_or, {input_or: [rot180_image]})[0]
        out_images.append(reverse_rotate_image_180(res)) 
        
        # flip 180 
        res = sess_or.run(output_or, {input_or: [flip_image(rot180_image)]})[0]
        out_images.append(reverse_rotate_image_180(reverse_flip_image(res)))          
       
        if use_mean:
            cnn_output.append(np.round(np.mean(np.array(out_images), axis=0)))
        else:
            cnn_output.append(np.round(np.median(np.array(out_images), axis=0)))
        
    cnn_output = np.array(cnn_output)
    cnn_output = np.transpose(cnn_output, [2, 0, 1, 3])  
    ssim_cnn, psnr_cnn = compute_ssim_psnr_batch(cnn_output, original_image) 
 
    return ssim_cnn, psnr_cnn
 
def compute_performance_indeces(test_images_gt, test_images, checkpoint):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images)):
        # pdb.set_trace()
        ssim_cnn, psnr_cnn = predict(test_images[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn  
        num_images += test_images_gt[index].shape[0]
      
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))

def read_images(test_path):

    test_images_gt = read_all_directory_images_from_directory_test(test_path, 'original')
    test_images = read_all_directory_images_from_directory_test(test_path, add_to_path='input_')
    
    return test_images_gt, test_images
    
checkpoint = tf.train.latest_checkpoint(params.folder_data)    
# checkpoint = './data_ckpt/model.ckpt128'
use_mean = True

test_path = './data/test'  
test_images_gt, test_images = read_images(test_path)    
 
compute_performance_indeces(test_images_gt, test_images, checkpoint) 