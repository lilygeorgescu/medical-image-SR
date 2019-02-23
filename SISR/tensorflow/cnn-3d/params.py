import cv2 as cv

L1_LOSS = 1
L2_LOSS = 2
# cv.INTER_LINEAR 
# cv.INTER_CUBIC
# cv.INTER_LANCZOS4
# cv.INTER_NEAREST

scale = 3
folder_name = '00001_0004'
folder_base_name = '3d-images'
interpolation_method = cv.INTER_LANCZOS4 
num_epochs = 150 
LOSS = L2_LOSS
learning_rate = 1e-3
dim_patch = 128
dim_depth = 16
kernel_size = 3
min_scale = 0.8
image_ext = 'png'
layers = 8

def show_params():
	print('\n\n\n\n')
	print('The configuration file is:')
	print('scale = {} '.format(scale))
	print('folder base name = {} '.format(folder_base_name))
	print('folder name = {} '.format(folder_name))
	print('image extension = {} '.format(image_ext))
	print('interpolation method = {} '.format(interpolation_method))
	print('num epochs = {} '.format(num_epochs))
	print('loss = {} '.format(LOSS))
	print('learning rate = {} '.format(learning_rate))
	print('dim patch = {} '.format(dim_patch))
	print('dim depth = {} '.format(dim_depth))
	print('kernel size = {} '.format(kernel_size))
	print('\n\n\n\n')