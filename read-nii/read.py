import nibabel as nib
import numpy as np
import pdb
import cv2 as cv
import os

def create_folders(directory_name):  
    if not os.path.exists(directory_name):
       os.makedirs(directory_name)
       print('directory created: {} '.format(directory_name)) 
    else:
       print('directory {} exists '.format(directory_name))


base_dir = 'nii'
filenames = os.listdir(base_dir)

# Get nibabel image object
for filename in filenames
    img = nib.load(os.path.join(base_dir, filename))
    
    folder_name = os.path.join('images', filename[:10])
    create_folders(folder_name)
    # Get data from nibabel image object (returns numpy memmap object)
    img_data = img.get_data()

    # Convert to numpy ndarray (dtype: uint16)
    img_data_arr = np.asarray(img_data)

    for i in range(img_data_arr.shape[2]):
        filename = 'image/%.5d.png' % i 
        cv.imwrite(os.path.join(folder_name, filename), img_data_arr[:, :, i]/img_data_arr[:, :, i].max() * 255)
 