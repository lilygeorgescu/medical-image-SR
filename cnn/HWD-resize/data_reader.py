import numpy as np
import utils
import params
from sklearn.utils import shuffle
import cv2 as cv 
import random 
import pdb

class DataReader:
    
    
    def __init__(self, train_path, eval_path, test_path, is_training=True, SHOW_IMAGES=False): 
    
        self.rotation_degrees = [0, 90, 180, 270]
        self.SHOW_IMAGES = SHOW_IMAGES        
        if(is_training):
            self.train_images, min_dim, min_depth = utils.read_all_directory_images_from_directory_as_list(train_path)
            self.train_images_depth_resize = utils.resize_depth_list_of_3d_image_standard(self.train_images, scale = 1/params.scale)
            self.dim_patch = min(params.dim_patch, min_dim)
            self.dim_depth = min(params.dim_depth, min_depth)
            self.train_images = shuffle(self.train_images) 
            self.num_train_images = len(self.train_images)
            self.index_train = 0 
            print('number of train images is %d' % (self.num_train_images))  
            print('dim patch is %d dim depth %d' % (self.dim_patch, self.dim_depth))
        
        else:
            self.test_images_gt, _, _ = utils.read_all_directory_images_from_directory_as_list(test_path)
            self.test_images = utils.resize_list_of_3d_image_standard(self.test_images_gt, scale = 1/params.scale)
            self.eval_images_gt, _, _ = utils.read_all_directory_images_from_directory_as_list(eval_path)
            self.eval_images = utils.resize_list_of_3d_image_standard(self.eval_images_gt, scale = 1/params.scale)  
            self.num_eval_images = len(self.eval_images)
            self.num_test_images = len(self.test_images)
            print('number of eval images is %d' % (self.num_eval_images))
            print('number of test images is %d' % (self.num_test_images))
        
        
    def get_next_batch_train(self, iteration):
     
        if(iteration == 0): # because we use only full batch
            self.index_train = 0 
            self.train_images = shuffle(self.train_images) 
        input_images = np.zeros((int(self.dim_depth / params.scale), int(self.dim_patch / params.scale), int(self.dim_patch / params.scale), params.num_channels))
        output_images = np.zeros((self.dim_depth, self.dim_patch, self.dim_patch, params.num_channels))
       
        image = self.train_images[self.index_train]  
        im_D = image.shape[0]
        im_H = image.shape[1]
        im_W = image.shape[2] 
        i = random.randint(0, im_H - self.dim_patch)
        j = random.randint(0, im_W - self.dim_patch)        
        k = random.randint(0, im_D - self.dim_depth)   
        output_images[:,:,:, 0] = image[k:k+self.dim_depth, i:i+self.dim_patch, j:j+self.dim_patch, 0]
        
        input_images = utils.resize_3d_image_standard(output_images, int(self.dim_depth/params.scale), int(self.dim_patch / params.scale), int(self.dim_patch / params.scale), interpolation_method=cv.INTER_NEAREST)
        output_h_w = utils.resize_depth_3d_image_standard(output_images, int(self.dim_depth/params.scale), output_images.shape[1], output_images.shape[2], interpolation_method=cv.INTER_NEAREST)
        if(self.SHOW_IMAGES):
            for image in output_images: 
                cv.imshow('output', image/255)
                cv.waitKey(0)
            for image in input_images:
                cv.imshow('input', image/255)
                cv.waitKey(0)
            
        self.index_train += 1
        return input_images, output_images, output_h_w
 
# data_reader = DataReader('./data/train', './data/validation', './data/test', SHOW_IMAGES=True)
# data_reader.get_next_batch_train(1)