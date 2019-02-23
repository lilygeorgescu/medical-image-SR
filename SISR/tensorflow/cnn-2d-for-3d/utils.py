import numpy as np
import cv2 as cv
from PIL import Image
import params as params
import os
import glob
import numpy as np
from skimage.measure import compare_ssim as ssim_sk
import math 
import pdb

SHOW_IMAGES = False

def ssim(img1, img2): 
    if(img1.shape[2] == 1):
        return ssim_sk(np.squeeze(img1), np.squeeze(img2))
    return ssim_sk(img1, img2)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def rotate(img, angle):
	num_rows, num_cols = img.shape[:2]
	rotation_matrix = cv.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
	img_rotation = cv.warpAffine(img, rotation_matrix, (num_cols, num_rows)) 
	return img_rotation

def get_output_directory_name():    
    return os.path.join('output-images', params.folder_name, str(params.scale)) + 'x'
    
def create_folders(): 
    directory_name = get_output_directory_name()
    if not os.path.exists(directory_name):
       os.makedirs(directory_name)
       print('directory created: {} '.format(directory_name)) 
    else:
       print('directory {} exists '.format(directory_name))
       
def read_all_images_from_directory():
    '''
        This function reads the images from the directory specified in params.py.
        The output is a numpy ndarray of size (num_images, height, width, channels).
    '''
    if not os.path.exists(params.folder_base_name):
        print('Error!! Folder base name does not exit')
    if not os.path.exists(os.path.join(params.folder_base_name, params.folder_name)):
        print('Error!! Folder name does not exit')  
        
    images_path = os.path.join(params.folder_base_name, params.folder_name,'*' + params.image_ext) 
    files = glob.glob(images_path)
    num_images = len(files)
    print('There are {} images in {}'.format(num_images, images_path))
    # read the first image to get the size of the images
    image = cv.imread(files[0], cv.IMREAD_GRAYSCALE)
    print('The size of the first image is {}'.format(image.shape))
    images = np.zeros((num_images, image.shape[0], image.shape[1], 1))
    images[0, :, :, 0] = image
    for index in range(1, num_images): 
        image = cv.imread(files[index], cv.IMREAD_GRAYSCALE)
        images[index, :, :, 0] = image 
        if(SHOW_IMAGES): 
            cv.imshow('image', image)
            cv.waitKey(0)
        
    return images
    
def resize_height_width_3d_image_standard(images, new_heigth, new_width, interpolation_method = cv.INTER_LINEAR):  
    '''
        images is a nd-array of size (num_images, new_heigth, new_width, num_channels) and this function resize every image from the input.
    '''
    num_images = images.shape[0] 
    num_channels = images.shape[-1]
    resized_images = np.zeros((num_images, new_heigth, new_width, num_channels))
 
    for index in range(num_images):
        image = images[index, :, :, :]
        if(num_channels == 1):
            resized_images[index, :, :, 0] = cv.resize(image, (new_width, new_heigth), interpolation = interpolation_method)
        else:
            resized_images[index, :, :, :] = cv.resize(image, (new_width, new_heigth), interpolation = interpolation_method)
            
        if(SHOW_IMAGES):
            cv.imshow('image',  resized_images[index, :, :, :] / 255)
            cv.waitKey(0)  
    
    return resized_images
    
def resize_depth_3d_image_standard(images, new_depth, heigth, width, interpolation_method = cv.INTER_LINEAR):
    '''
        Resize the depth of the 3d image by taking every coord as a depth row and resizing it.
    '''
    resized_3d_images = np.zeros((new_depth, heigth, width, images.shape[3]))         
    for y in range(heigth):
        for x in range(width):
            depth_row = images[:, y, x]
            resized_depth_row = cv.resize(depth_row, (1, new_depth), interpolation = interpolation_method)
            resized_3d_images[:, y, x, 0] = resized_depth_row.ravel()
            
    if(SHOW_IMAGES):    
        for index in range(new_depth):
            image = resized_3d_images[index, :, :, :] 
            cv.imshow('image', image / 255)
            cv.waitKey(0)            
     
    return resized_3d_images
    
def resize_3d_image_standard(images, new_depth, new_heigth, new_width, interpolation_method = cv.INTER_LINEAR): 
     
    resized_images = resize_height_width_3d_image_standard(images, new_heigth, new_width, interpolation_method)
    resized_3d_images = resize_depth_3d_image_standard(resized_images, new_depth, new_heigth, new_width, interpolation_method)
       
    return resized_3d_images
     
def write_3d_images(images, prefix):
    '''
        This function writes the images in the directory specified in params.py with the prefix specified as a param.
        The input is a numpy ndarray of size (num_images, height, width, channels) and a string.
    '''
    num_images = images.shape[0]
    directory_name = os.path.join(get_output_directory_name(), prefix)
    if not os.path.exists(directory_name):
       os.makedirs(directory_name)
       print('directory created: {} '.format(directory_name)) 
    for index in range(num_images):
        image = images[index, :, :, :] 
        cv.imwrite(os.path.join(directory_name, str(index) + '.' + params.image_ext), image)
    

def flip_images(images):
    num_images = images.shape[0]   
    flipped_images = np.zeros(images.shape)
    for index in range(num_images):
        image = images[index, :, :, 0]  
        flipped_images[index, :, :, 0] = cv.flip(image, 1)
        
    return flipped_images
    
def rotate_images(images, angle):
    num_images = images.shape[0]   
    rotated_images = np.zeros(images.shape)
    for index in range(num_images):
        image = images[index, :, :, 0]  
        rotated_images[index, :, :, 0] = rotate(image, angle)
        
    return rotated_images
    