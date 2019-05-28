from utils import *
import params
import tensorflow as tf
import pdb
import cv2 as cv 

def trim_image(image):
    image[image > 255] = 255
    return image

def predict_1_2(downscaled_image, checkpoint):
 
    # network for original image
    config = tf.ConfigProto(
            device_count = {'GPU': 1}
        ) 
    
    sess_or = tf.Session(config=config)
    with sess_or:
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        _, output_or = params.network_architecture(input_or) 
        
    sess_tr = tf.Session(config=config)
    with sess_tr:
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        _, output_tr = params.network_architecture(input_tr, reuse=True)     
     
    saver = tf.train.Saver()
    print('restoring from ' + checkpoint)
    saver.restore(sess_or, checkpoint)
    saver.restore(sess_tr, checkpoint)
    
    num_images = downscaled_image.shape[0]
    cnn_output = []
    for image in downscaled_image:
        out_images = [] 
        
        # original 0 
        res = trim_image(sess_or.run(output_or, {input_or: [image]})[0])
        out_images.append(res)
        
        # flip 0 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(image)]})[0])
        out_images.append(reverse_flip_image(res))
         
        # original 180
        rot180_image = rotate_image_180(image)
        res = trim_image(sess_or.run(output_or, {input_or: [rot180_image]})[0])
        out_images.append(reverse_rotate_image_180(res)) 
        
        # flip 180 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(rot180_image)]})[0])
        out_images.append(reverse_rotate_image_180(reverse_flip_image(res)))          
       
        if use_mean:
            cnn_output.append(np.round(np.mean(np.array(out_images), axis=0)))
        else:
            cnn_output.append(np.round(np.median(np.array(out_images), axis=0)))
        
    cnn_output = np.array(cnn_output)
    cnn_output = np.transpose(cnn_output, [2, 0, 1, 3])   
 
    return cnn_output

def predict_2_1(downscaled_image, checkpoint):
 
    # network for original image
    config = tf.ConfigProto(
            device_count = {'GPU': 1}
        ) 
    
    sess_or = tf.Session(config=config)
    with sess_or:
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        _, output_or = params.network_architecture(input_or) 
        
    sess_tr = tf.Session(config=config)
    with sess_tr:
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        _, output_tr = params.network_architecture(input_tr, reuse=True)     
     
    saver = tf.train.Saver()
    print('restoring from ' + checkpoint)
    saver.restore(sess_or, checkpoint)
    saver.restore(sess_tr, checkpoint)
    
    num_images = downscaled_image.shape[0]
    cnn_output = []
    for image in downscaled_image:
        out_images = [] 
        
        # original 0 
        res = trim_image(sess_or.run(output_or, {input_or: [image]})[0])
        out_images.append(res)
        
        # flip 0 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(image)]})[0])
        out_images.append(reverse_flip_image(res))
         
        # original 180
        rot180_image = rotate_image_180(image)
        res = trim_image(sess_or.run(output_or, {input_or: [rot180_image]})[0])
        out_images.append(reverse_rotate_image_180(res)) 
        
        # flip 180 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(rot180_image)]})[0])
        out_images.append(reverse_rotate_image_180(reverse_flip_image(res)))          
       
        if use_mean:
            cnn_output.append(np.round(np.mean(np.array(out_images), axis=0)))
        else:
            cnn_output.append(np.round(np.median(np.array(out_images), axis=0)))
        
    cnn_output = np.array(cnn_output)
    cnn_output = np.transpose(cnn_output, [2, 1, 0, 3])   
 
    return cnn_output
    
def read_images(test_path):
 
    add_to_path = 'original' 
    test_images_gt = read_all_directory_images_from_directory_test(test_path, add_to_path=add_to_path)
    test_images_1_2 = read_all_directory_images_from_directory_test(test_path, add_to_path='input_')    
    test_images_2_1 = read_all_directory_images_from_directory_test(test_path, add_to_path='input_2_1')  

    return test_images_gt, test_images_1_2, test_images_2_1



        
def predict(downscaled_image_1_2, downscaled_image_2_1, original_image, checkpoint): 

    scale_factor = params.scale    
    standard_resize = resize_height_width_3d_image_standard(downscaled_image_1_2, int(downscaled_image_1_2.shape[1]), int(downscaled_image_1_2.shape[2])*scale_factor, interpolation_method = params.interpolation_method) 
    
    image_1_2 = predict_1_2(downscaled_image_1_2, checkpoint)
    tf.reset_default_graph()
    image_2_1 = predict_2_1(downscaled_image_2_1, checkpoint) 
    standard_resize = np.transpose(standard_resize, [2, 0, 1, 3]) 
    cnn_output = 0.5 * (image_1_2 + image_2_1)
         
          
    ssim_cnn, psnr_cnn = compute_ssim_psnr_batch(cnn_output, original_image)
    ssim_standard, psnr_standard = compute_ssim_psnr_batch(standard_resize, original_image)

    return ssim_cnn, psnr_cnn, ssim_standard, psnr_standard 
    
    
def compute_performance_indeces(test_images_gt, test_images_1_2, test_images_2_1, checkpoint):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images_gt)):  
            
        ssim_cnn, psnr_cnn, ssim_standard, psnr_standard = predict(test_images_1_2[index], test_images_2_1[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images_gt[index].shape[0]
     
    print('standard {} --- psnr = {} ssim = {}'.format(test_path, psnr_standard_sum/num_images, ssim_standard_sum/num_images)) 
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))
    

 
    
# checkpoint = tf.train.latest_checkpoint(params.folder_data)    
checkpoint = './data_ckpt/model.ckpt13'
use_mean = True

test_path = './data/test'   
test_images_gt, test_images_1_2, test_images_2_1 = read_images(test_path)  
 
compute_performance_indeces(test_images_gt, test_images_1_2, test_images_2_1, checkpoint) 