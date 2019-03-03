import cv2 as cv

image_name = '61962566.png';
im = cv.imread(image_name)
scale = 2
# cv.INTER_LINEAR 
# cv.INTER_CUBIC
# cv.INTER_LANCZOS4
# cv.INTER_NEAREST
new_width = int(im.shape[1] / scale)
new_heigth = int(im.shape[0] / scale)
im_linear = cv.resize(im, (new_width, new_heigth), interpolation = cv.INTER_LINEAR)
im_cubic = cv.resize(im, (new_width, new_heigth), interpolation = cv.INTER_CUBIC)
im_lanc = cv.resize(im, (new_width, new_heigth), interpolation = cv.INTER_LANCZOS4)
im_nearest = cv.resize(im, (new_width, new_heigth), interpolation = cv.INTER_NEAREST)
cv.imshow('im', im)
cv.imshow('im_linear', im_linear)
cv.imshow('im_cubic', im_cubic)
cv.imshow('im_lanc', im_lanc)
cv.imshow('im_nearest', im_nearest)
cv.waitKey(0)