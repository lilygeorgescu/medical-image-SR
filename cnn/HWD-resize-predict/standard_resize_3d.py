import utils
import cv2 as cv
import numpy as np
import pdb

def resize(downscaled_image, original_image, interpolation_method): 
 
    standard_resize = utils.resize_3d_image_standard(downscaled_image, new_depth=original_image.shape[0], new_height=original_image.shape[1], new_width=original_image.shape[2], interpolation_method=interpolation_method) 
      
    ssim_standard, psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image) 
    
    return ssim_standard, psnr_standard 
 
    
def compute_performance_indeces(test_images_gt, test_images, interpolation_method): 
    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images)): 
        ssim_standard, psnr_standard = resize(test_images[index], test_images_gt[index], interpolation_method) 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images_gt[index].shape[0] 
                
    return psnr_standard_sum/num_images, ssim_standard_sum/num_images 
  
def read_images(test_path): 
    if use_hw_d:
        add_to_path = 'input_hw_d_%d' % scale_factor
    else:
        add_to_path = 'input'
        
    test_images_gt = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='original')
    test_images = utils.read_all_directory_images_from_directory_test(test_path, add_to_path=add_to_path)
    
    return test_images_gt, test_images    

use_hw_d = True    
test_path = './data/test'  
scale_factor = 4
test_images_gt, test_images = read_images(test_path)


    
interpolation_methods= {'INTER_LINEAR': cv.INTER_LINEAR,
                        'INTER_CUBIC': cv.INTER_CUBIC,
                        'INTER_LANCZOS4': cv.INTER_LANCZOS4,
                        'INTER_NEAREST': cv.INTER_NEAREST}

for interpolation_method in interpolation_methods.keys():
    psnr, ssim = compute_performance_indeces(test_images_gt, test_images, interpolation_methods[interpolation_method])
    print('interpolation method %s has ssim %f psnr %f' % (interpolation_method, ssim, psnr))

    
 