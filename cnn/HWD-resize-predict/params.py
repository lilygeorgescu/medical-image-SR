import cv2 as cv 
scale = 2 
# cv.INTER_LINEAR 
# cv.INTER_CUBIC
# cv.INTER_LANCZOS4
# cv.INTER_NEAREST
interpolation_method = cv.INTER_LANCZOS4  
layers = 8 

scale = 2
folder_name = '00001_0004'
folder_base_name = '../cnn-3d/3d-images'
# cv.INTER_LINEAR 
# cv.INTER_CUBIC
# cv.INTER_LANCZOS4
# cv.INTER_NEAREST
interpolation_method = cv.INTER_LANCZOS4 
num_epochs = 50
LOSS = 1
learning_rate = 1e-4
dim_patch = '_14_28' 
kernel_size = 5 
image_ext = 'png'
folder_data = './data_ckpt/'
layers = 8
num_channels = 1
tf_version = 1.02
ckpt_name = 'model.ckpt'
latest_ckpt_filename = 'latest_epoch_tested'