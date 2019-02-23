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
            self.train_images, min_dim = utils.read_all_directory_images_from_directory(train_path)
            self.dim_patch = min(params.dim_patch, min_dim)
            self.train_images = shuffle(self.train_images) 
            self.num_train_images = len(self.train_images)
            self.index_train = 0 
            print('number of train images is %d' % (self.num_train_images))  
            print('dim patch is %d' % self.dim_patch)
        
        else:
            self.test_images_gt = utils.read_all_directory_images_from_directory_test(test_path)
            self.test_images = utils.resize_list_of_3d_image_standard(self.test_images_gt, scale = 1/params.scale)
            self.eval_images_gt = utils.read_all_directory_images_from_directory_test(eval_path)
            self.eval_images = utils.resize_list_of_3d_image_standard(self.eval_images_gt, scale = 1/params.scale)  
            self.num_eval_images = len(self.eval_images)
            self.num_test_images = len(self.test_images)
            print('number of eval images is %d' % (self.num_eval_images))
            print('number of test images is %d' % (self.num_test_images))
        
        
    def get_next_batch_train(self, iteration, batch_size = 32):
    
        end = self.index_train + batch_size 
        if(iteration == 0): # because we use only full batch
            self.index_train = 0
            end = batch_size 
            self.train_images = shuffle(self.train_images) 
        input_images = np.zeros((batch_size, int(self.dim_patch / params.scale), int(self.dim_patch / params.scale), params.num_channels))
        output_images = np.zeros((batch_size, self.dim_patch, self.dim_patch, params.num_channels))
        start = self.index_train
        for idx in range(start, end): 
            image = self.train_images[idx]  
            im_H = image.shape[0]
            im_W = image.shape[1] 
            i = random.randint(0, im_H - self.dim_patch )
            j = random.randint(0, im_W - self.dim_patch ) 
            
            image_patch = image[i:i + self.dim_patch , j:j + self.dim_patch ]
            # augumentation
            idx_degree = random.randint(0, len(self.rotation_degrees) - 1) 
            image_patch = utils.rotate(image_patch, self.rotation_degrees[idx_degree])
            
            if(params.num_channels == 1): 
                input_images[idx - start, :, :, 0] = cv.resize(image_patch, (0,0), fx = 1.0 / params.scale, fy = 1.0 / params.scale)  
                output_images[idx - start, :, :, 0] =  image_patch       
            else:
                input_images[idx - start, :, :, :] = cv.resize(image_patch, (0,0), fx = 1.0 / params.scale, fy = 1.0 / params.scale)
                output_images[idx - start, :, :, :] = image_patch
                
            if(self.SHOW_IMAGES):
                cv.imshow('input', input_images[idx - start, :, :, :]/255)
                cv.imshow('output', output_images[idx - start, :, :, :]/255)
                cv.waitKey(0)
        
        self.index_train = end
        return input_images, output_images
 
        