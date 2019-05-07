import numpy as np
import cv2 as cv

image_path = 'bird.png'
image = cv.imread(image_path, 0)

cv.imshow('image', image)
cv.waitKey(0)

rotated_image = np.rot90(image)
cv.imshow('rot 90', rotated_image)
cv.waitKey(0)

rotated_image = np.rot90(rotated_image)
cv.imshow('rot 180', rotated_image)
cv.waitKey(0)