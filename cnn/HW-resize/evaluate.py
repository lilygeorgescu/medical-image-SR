from utils import *
import params
import tensorflow as tf
import pdb
import cv2 as cv

def predict(downscaled_image=None, original_image=None, path_images=None, path_original_images=None):
  
    if path_images is None and downscaled_image is None:
            raise ValueError('if images is None path_images must not be none.') 
    if path_original_images is None and original_image is None:
            raise ValueError('if path_original_images is None original_image must not be none.')
            
    if original_image is None:
        original_image = read_all_images_from_directory(path_original_images)
     
    mean = np.loadtxt('mean.txt')
    if downscaled_image is None:
        downscaled_image = read_all_images_from_directory(path_images)
    downscaled_image = downscaled_image / 255  
    downscaled_image = downscaled_image - mean
    
    # network for original image
    config = tf.ConfigProto(
            device_count = {'GPU': 0}
        ) 
    
    sess_or = tf.Session(config=config)
    with sess_or:
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        output_or = params.network_architecture(input_or) 
    # pdb.set_trace()    
    sess_tr = tf.Session(config=config)
    # with sess_tr:
        # input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        # output_tr = params.network_architecture(input_tr, reuse=True)     
    
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
        
        # original 90
        rot90_image = rotate_image_90(image)
        res = sess_tr.run(output_tr, {input_tr: [rot90_image]})[0]
        out_images.append(reverse_rotate_image_90(res)) 
        
        # flip 90 
        res = sess_tr.run(output_tr, {input_or: [flip_image(rot90_image)]})[0]
        out_images.append(reverse_rotate_image_90(reverse_flip_image(res)))   

        # original 180
        rot180_image = rotate_image_180(image)
        res = sess_or.run(output_or, {input_or: [rot180_image]})[0]
        out_images.append(reverse_rotate_image_180(res)) 
        
        # flip 180 
        res = sess_or.run(output_or, {input_or: [flip_image(rot180_image)]})[0]
        out_images.append(reverse_rotate_image_180(reverse_flip_image(res)))          
        
        # original 270
        rot270_image = rotate_image_270(image)
        res = sess_tr.run(output_tr, {input_tr: [rot270_image]})[0]
        out_images.append(reverse_rotate_image_270(res)) 
        
        # flip 270 
        res = sess_tr.run(output_tr, {input_tr: [flip_image(rot270_image)]})[0]
        out_images.append(reverse_rotate_image_270(reverse_flip_image(res)))           
        
        cnn_output.append(np.mean(np.array(out_images), axis=0))
        # cnn_output.append(np.median(np.array(out_images), axis=0))
    ssim_cnn, psnr_cnn = utils.compute_ssim_psnr(cnn_output, original_image, stride=stride) 
 
    print('cnn --- psnr = {} ssim = {}'.format(psnr_cnn, ssim_cnn))  

    if(path_images != None):
        write_3d_images(path_images, cnn_output, 'cnn')
        
        
predict(path_images='./data/train_/set5/input_', path_original_images='./data/train_/set5/') 