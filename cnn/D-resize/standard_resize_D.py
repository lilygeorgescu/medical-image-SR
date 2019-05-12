import utils
import params
import cv2 as cv
import pdb
import numpy as np

def resize(downscaled_image, original_image, interpolation_method): 
    
    standard_resize = utils.resize_height_width_3d_image_standard(downscaled_image, int(downscaled_image.shape[1]), int(downscaled_image.shape[2]*2), interpolation_method=interpolation_method)
    
    if use_original: 
        standard_resize = np.transpose(standard_resize, [2, 0, 1, 3])  
    
    ssim_standard, psnr_standard = utils.compute_ssim_psnr_batch(standard_resize, original_image)

    return ssim_standard, psnr_standard 
        
def read_images(test_path):

    if use_original:
        add_to_path = 'original'
    else:
        add_to_path = 'transposed'
        
    test_images_gt = utils.read_all_directory_images_from_directory_test(test_path, add_to_path=add_to_path)
    test_images = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='input_')
    
    return test_images_gt, test_images
     
def compute_performance_indeces(test_images_gt, test_images, interpolation_method): 
    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images)): 
        ssim_standard, psnr_standard = resize(test_images[index], test_images_gt[index], interpolation_method) 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images_gt[index].shape[0]
         
            
    return psnr_standard_sum/num_images, ssim_standard_sum/num_images 
    
interpolation_methods= {'INTER_LINEAR': cv.INTER_LINEAR,
                        'INTER_CUBIC': cv.INTER_CUBIC,
                        'INTER_LANCZOS4': cv.INTER_LANCZOS4,
                        'INTER_NEAREST': cv.INTER_NEAREST}
 
test_path = './data/test' 

use_original = False
test_images_gt, test_images = read_images(test_path)

for interpolation_method in interpolation_methods.keys():
    psnr, ssim = compute_performance_indeces(test_images_gt, test_images, interpolation_methods[interpolation_method])
    print('interpolation method %s has ssim %f psnr %f' % (interpolation_method, ssim, psnr))
    
# tranposed images    
# interpolation method INTER_LINEAR has ssim 0.864469 psnr 36.881867
# interpolation method INTER_NEAREST has ssim 0.875036 psnr 37.078668
# interpolation method INTER_CUBIC has ssim 0.885053 psnr 37.163100
# interpolation method INTER_LANCZOS4 has ssim 0.888183 psnr 37.231696

# original images
# interpolation method INTER_NEAREST has ssim 0.903308 psnr 34.100390
# interpolation method INTER_LINEAR has ssim 0.897913 psnr 33.909730
# interpolation method INTER_LANCZOS4 has ssim 0.912667 psnr 34.280864
# interpolation method INTER_CUBIC has ssim 0.910502 psnr 34.196829
    