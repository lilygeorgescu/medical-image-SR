import nibabel as nib
import numpy as np
import pdb
import cv2 as cv

# Get nibabel image object
img = nib.load("IXI017-Guys-0698-IXIT2WTS_-s231_-0401-00004-000001-01.nii")

# Get data from nibabel image object (returns numpy memmap object)
img_data = img.get_data()

# Convert to numpy ndarray (dtype: uint16)
img_data_arr = np.asarray(img_data)

for i in range(img_data_arr.shape[2]):
    filename = 'image/%.5d.png' % i 
    cv.imwrite(filename, img_data_arr[:, :, i]/img_data_arr[:, :, i].max() * 255)

pdb.set_trace()