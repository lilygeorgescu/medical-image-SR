import utils
import params
import cv2 as cv
import pdb

def predict(images=None, path_images=None, scale_factor=params.scale, interpolation_method=params.interpolation_method): 

    if(type(images) != type(None)):
        original_image = images
    else:
        if(path_images == None):
            raise ValueError('if images is None path_images must not be none.')
        original_image = utils.read_all_images_from_directory(path_images)   
     
    # standard resize 
    downscaled_image = utils.resize_height_width_3d_image_standard(original_image,int(original_image.shape[1] / scale_factor), int(original_image.shape[2] / scale_factor))
    standard_resize = utils.resize_height_width_3d_image_standard(downscaled_image, int(original_image.shape[1]), int(original_image.shape[2]), interpolation_method=interpolation_method) 
    
    num_images = standard_resize.shape[0]
    sum_ssim_standard = 0 
    sum_psnr_standard = 0 

    for index in range(num_images):  
        psnr_standard = utils.psnr(original_image[index, :, :], standard_resize[index, :, :])
        ssim_standard = utils.ssim(original_image[index, :, :], standard_resize[index, :, :]) 
             
        sum_ssim_standard += ssim_standard
        sum_psnr_standard += psnr_standard  
    # print('standard --- psnr = {} ssim = {}'.format(sum_psnr_standard, sum_ssim_standard))  
    return sum_psnr_standard, sum_ssim_standard, num_images
    

list_images = utils.read_all_directory_images_from_directory_test('./data/test')    
sum_ssim_standard = 0 
sum_psnr_standard = 0 
num_images_all = 0
interpolation_method = cv.INTER_LANCZOS4

for images in list_images:
    psnr_standard, ssim_standard, num_images = predict(images=images, interpolation_method=interpolation_method)
    sum_ssim_standard += ssim_standard
    sum_psnr_standard += psnr_standard    
    num_images_all += num_images
    
print('{} standard --- psnr = {} ssim = {}'.format(interpolation_method, sum_psnr_standard / num_images_all, sum_ssim_standard / num_images_all))  