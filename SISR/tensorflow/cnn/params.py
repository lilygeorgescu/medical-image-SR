import cv2 as cv

L1_LOSS = 1
L2_LOSS = 2
#cv.INTER_LINEAR 
#cv.INTER_CUBIC
#cv.INTER_LANCZOS4
#cv.INTER_NEAREST

scale = 2
image_name = '61963414.png'
interpolation_method = cv.INTER_LANCZOS4 
num_epochs = 1500 
LOSS = L1_LOSS
learning_rate = 5e-3
dim_patch = 128
kernel_size = 3
min_scale = 0.9
tf_version = 1.02
layers = 5
num_channels = 1
folder_data = './data/'
def show_params():
	print('\n\n\n\n')
	print('The configuration file is:')
	print('scale = {} '.format(scale))
	print('image name = {} '.format(image_name))
	print('interpolation method = {} '.format(interpolation_method))
	print('num epochs = {} '.format(num_epochs))
	print('loss = {} '.format(LOSS))
	print('learning rate = {} '.format(learning_rate))
	print('dim patch = {} '.format(dim_patch))
	print('kernel size = {} '.format(kernel_size))
	print('\n\n\n\n')