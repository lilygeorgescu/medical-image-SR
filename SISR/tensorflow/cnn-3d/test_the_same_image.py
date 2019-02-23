import cv2 as cv
import numpy as np
import pdb

path_1 = 'D:/disertatie/SISR/tensorflow/cnn-3d/output-images/00001_0004/1x/test_cnn/1.png'
path_2 = 'D:/disertatie/SISR/tensorflow/cnn-3d/output-images/00001_0004/1x/cnn/1.png'

img_1 = cv.imread(path_1, cv.IMREAD_GRAYSCALE) 
img_2 = cv.imread(path_2, cv.IMREAD_GRAYSCALE)

diff = abs(np.float32(img_1) - np.float32(img_2)) 
 
print('the maximum diff is {} .'.format(diff.max()))