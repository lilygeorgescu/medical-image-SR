from utils import *
import params

def predict(downscaled_image=None, original_image=None, path_images=None, path_original_images=None):
  
    if path_images is None and downscaled_image is None:
            raise ValueError('if images is None path_images must not be none.') 
    if path_original_images is None and original_image is None:
            raise ValueError('if path_original_images is None original_image must not be none.')
            
    if original_image is None:
        original_image = utils.read_all_images_from_directory(path_original_images)
     
    mean = np.loadtxt('mean.txt')
    if downscaled_image is None:
        downscaled_image = utils.read_all_images_from_directory(path_images)
    downscaled_image = downscaled_image / 255  
    downscaled_image = downscaled_image - mean
    
    # network for original image
    config = tf.ConfigProto(
            device_count = {'GPU': 1}
        ) 
    
    sess_or = tf.Session(config=config)
    with sess_or:
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        output_or = params.network_architecture(input_or, params.kernel_size) 
        
    sess_tr = tf.Session(config=config)
    with sess_tr:
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        output_tr = params.network_architecture(input_tr, params.kernel_size)     


 

predict(path_images='./data/train_/00001_0003/input_', path_original_images='./data/train_/00001_0003/', write_images=False) 