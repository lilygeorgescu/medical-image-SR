import numpy as np
import cv2 as cv
from PIL import Image
import imutils
from skimage.measure import compare_ssim as ssim_sk
import math 
import pdb

def rotate(img, angle):
	# num_rows, num_cols = img.shape[:2]
	# rotation_matrix = cv.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
	# img_rotation = cv.warpAffine(img, rotation_matrix, (num_cols, num_rows)) 
	img_rotation = imutils.rotate(img, angle)
	return img_rotation
    
 

def ssim(img1, img2): 
    if(len(img1.shape) == 3):
        return ssim_sk(np.squeeze(img1), np.squeeze(img2))
    
    return ssim_sk(np.squeeze(img1), np.squeeze(img2), multichannel=(len(img1.shape) == 3))

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2) 
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
