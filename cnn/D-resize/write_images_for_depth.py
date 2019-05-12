import os
from utils_old import *
import pdb
import cv2 as cv

# base_dir = './data/train'
base_dir = './data/test'

folder_names = os.listdir(base_dir)
transposed_3d_images, _, _ = read_images_depth_training(os.path.join(base_dir))


for index_folder in range(len(folder_names)):
    print('%d/%d' % (index_folder, len(folder_names)))
    image_names = os.listdir(os.path.join(base_dir, folder_names[index_folder]))  
    create_folders(os.path.join(base_dir, folder_names[index_folder], 'original'))
    create_folders(os.path.join(base_dir, folder_names[index_folder], 'transposed')) 

      
    # save transposed images
    index_new_images = 0
    for image in transposed_3d_images[index_folder]: 
        cv.imwrite(os.path.join(base_dir, folder_names[index_folder], 'transposed' , '%.5d.' % index_new_images + params.image_ext), image)
        index_new_images += 1
        # move the images
    for image_name in image_names:
        os.rename(os.path.join(base_dir, folder_names[index_folder], image_name), os.path.join(base_dir, folder_names[index_folder], 'original', image_name))
    
    