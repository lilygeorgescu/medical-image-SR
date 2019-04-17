import numpy as np
import random
import pdb
from glob import glob
import pickle
import time
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib import slim

from lstm_baseline import LstmModel
from tensorflow import flags

from moviepy.editor import VideoFileClip

# import lycon
# FLAGS = flags.FLAGS
# flags.DEFINE_integer("batch_size", 10, "batch size")
# flags.DEFINE_integer("video_num_frames", 10,"")
# flags.DEFINE_integer("used_num_frames", 1,"")
# flags.DEFINE_string("ckpt_path", '/data/experimental/vision-group/captions/graph_conv_an/models2/model_noise1_conv_lr0001', "path to inception ckpt")




# read video.mp4

def read_video_file(filename,num_frames=10, H = 240, W = 320):
	#time1 = time.time()
	filename = filename.decode("utf-8") 
	clip = VideoFileClip(filename, target_resolution=(None, None), resize_algorithm='bilinear')
	width = clip.size[0]
	height = clip.size[1]

	ratio = width / height

	if (width / W) < (height / H):
		new_width = W
		new_height = int(new_width/ratio)
	else:
		new_height = H
		new_width = int(ratio * new_height)

	new_clip = clip.resize(width=new_width,height=new_height)

	time11 = time.time()
	#frames = list(new_clip.iter_frames(fps=new_clip.fps))
	frames = []
	random_start = np.random.rand() * (new_clip.duration / num_frames)
	time_frames = np.linspace(random_start,new_clip.duration,num_frames)
	#print(f'frames: {time_frames}')
	for t in time_frames:
		frames.append(new_clip.get_frame(t))
	frames = np.array(frames)

	
	real_height = frames.shape[1] 
	real_width = frames.shape[2]
	#time11 = time.time()
	bigger_frames = np.zeros([frames.shape[0], new_height, new_width, 3], dtype=np.uint8)
	if np.random.rand() < 0.5:
		bigger_frames[:,:real_height, :real_width] = frames
	else:
		bigger_frames[:,:real_height, :real_width] = frames[:,::,::-1]

	return bigger_frames


def random_resize_to_SD(image, H=240, W=320):
	height = image.shape[1]
	width = image.shape[2]
	if (width / W) < (height / H) :
		max_to_crop = height-H
		min_to_crop = 0
		to_crop = np.random.randint(min_to_crop, max_to_crop+1)
		image_cropped = image[:,to_crop:to_crop+H,:]

	else:
		max_to_crop = width-W
		min_to_crop = 0
		to_crop = np.random.randint(min_to_crop, max_to_crop+1)
		image_cropped = image[:,:,to_crop:to_crop+W]
	return image_cropped


def _parse_function_py(filepath, label):
	video = np.array(read_video_file(filepath))
	video = random_resize_to_SD(video)
	return video, label

def _parse_function_tf(filepath, label):
	return tf.py_func(_parse_function_py, [filepath, label], [tf.uint8, tf.int32])

def _parse_aug(filepath, label,num_classes=600):
	video,label = _parse_function_tf(filepath, label)
	video = random_augmentation(video)
	label = tf.one_hot(label,num_classes)
	return video, label

def _parse_nonaug(filepath, label,num_classes=600):
	video,label = _parse_function_tf(filepath, label)
	label = tf.one_hot(label,num_classes)
	return video, label

def random_augmentation(video):
	#video = tf.image.random_flip_up_down(video)
	video = tf.image.random_brightness(video, max_delta=0.3)
	return video

def get_labels(image_paths, gt_dict, mapping,aux=0):
	labels = []
	#pdb.set_trace
	for i,path in enumerate(image_paths):
		#video_name = path.split("/")[-1][:-18]#[6:-5]
		video_name = path.split("/")[-1][-(16+aux):-(5+aux)]
		if video_name not in gt_dict:
			print(f'can not find label for {path}')
		else:
			label = mapping[gt_dict[video_name]['annotations']['label']]
			labels.append(label)
	return labels

def get_labels_mnist(image_paths, gt_dict, mapping,aux=0):
	labels = []
	for i,path in enumerate(image_paths):
		#video_name = path.split("/")[-1][:-18]#[6:-5]
		video_name = path.split("/")[-1][-(16+aux):-(5+aux)]
		if video_name not in gt_dict:
			#pdb.set_trace()
			print(path)
		else:
			label = mapping[gt_dict[video_name]['annotations']['label']]
			labels.append(label)
	return labels
########################  READERS ###################################
def inception_preprocessing_mnist(images):
	return images
	
def inception_preprocessing_i3d_tf(images):
	images = tf.cast(images, tf.float32)
	images = tf.subtract(images, 114.75)
	images = tf.div(images, 57.375)
	return images
def inception_preprocessing_tf(images):
	images = tf.cast(images, tf.float32)
	images = tf.div(images, 127.5)
	images = tf.subtract(images, 1.)
	return images

def inception_preprocessing(images):
	# images = [image / 255. for image in images]
	# images = [image - 0.5 for image in images]
	# images = [image * 2 for image in images]

	images = images / 127.5
	images = images - 1.

	return images

def read_data_kinetics_files(dirs, gt_dict, num_classes, mapping, dir_path):
	# dirs contains a list of directories
	# each dir contains one video stored as images
	all_videos = np.zeros([len(dirs), 10, 240, 320, 3])
	all_labels = np.zeros([len(dirs), 600])
	for i,crt_dir in enumerate(dirs):
		x,y = read_data_kinetics(crt_dir, gt_dict, num_classes, mapping, dir_path)

		all_videos[i] = x[0]
		all_labels[i] = y[0]
	return all_videos, all_labels




def resize_to_SD(image):
	height = image.shape[0]
	width = image.shape[1]
	if (width/height) < (320/240):
		new_width = 320
		new_height = int(height * new_width / width)
		image_rescaled = lycon.resize(image, height=new_height, width=new_width)
		to_crop = int((new_height - height)/2)
		image_cropped = image_rescaled[to_crop:to_crop+240,:]
	else:
		to_crop = int((width - 320)/2)
		image_cropped = image[:,to_crop:to_crop+320]	
	return image_cropped

def read_video(dir):
	# read a video frame by frame from dir
	clip_images = glob(dir+"/*.jpeg")
	video_frames = []
	for i,image in enumerate(clip_images):
		time1=time.time()
		img = lycon.load(image)
		#time2=time.time()
		#img_resized = resize_to_SD(img)
		#time3=time.time()
		img_preprocess = inception_preprocessing(img)
		time4=time.time()
		# print("load time: ", time2-time1)
		# print("resize time: ", time3-time2)
		# print("inception time: ", time4-time3)

		video_frames.append(img_preprocess)
	video_frames = np.array(video_frames)
	#pdb.set_trace()
	video_frames = np.reshape(video_frames,[10,-1,video_frames.shape[2],video_frames.shape[3]])
	return video_frames

def read_video_old(dir):
	# read a video frame by frame from dir
	clip_images = glob(dir+"/*.jpeg")
	video_frames = []
	for i,image in enumerate(clip_images):
		time1=time.time()
		img = lycon.load(image)
		time2=time.time()
		img_resized = resize_to_SD(img)
		time3=time.time()
		img_preprocess = inception_preprocessing(img_resized)
		time4=time.time()
		# print("load time: ", time2-time1)
		# print("resize time: ", time3-time2)
		# print("inception time: ", time4-time3)

		video_frames.append(img_preprocess)
	video_frames = np.array(video_frames)
	return video_frames

def read_data_kinetics(dir_name, gt_dict, num_classes, mapping, dir_path):
	# dir contains 10 frames from one video
	batch_videos = []
	batch_labels = []
	
	video_dir = dir_path + "/" + dir_name
	video_name = dir_name
	time1=time.time()
	video = read_video(video_dir)
	time2=time.time()
	# print("read_video: ", time2-time1)
	batch_videos.append(video)
	# pdb.set_trace()
	label = mapping[gt_dict[video_name]['annotations']['label']]
	y = np.eye(num_classes)[label]
	batch_labels.append(y)
	
	# pdb.set_trace()
	return np.array(batch_videos), np.array(batch_labels)


# sample batch_size elements fromd data => (x,y)
def extract_sample(data, batch_size):
	sample_idx = random.sample(range(len(data[0])), batch_size)

	x = np.asarray([data[0][x] for x in sample_idx])
	y = np.asarray([data[1][x] for x in sample_idx])

	return x, y

def extract_sample_boxes(data, batch_size):
	sample_idx = random.sample(range(len(data[0]['videos'])), batch_size)

	x_videos = np.asarray([data[0]['videos'][x] for x in sample_idx])
	x_digits = np.asarray([data[0]['digits'][x] for x in sample_idx])
	x_digits_coord = np.asarray([data[0]['digits_coord'][x] for x in sample_idx])

	y = np.asarray([data[1][x] for x in sample_idx])

	x = {}
	x['videos'] = x_videos
	x['digits'] = x_digits
	x['digits_coord'] = x_digits_coord
	return x['videos'], x, y

def read_video0(filename, fps):
	clip = VideoFileClip(filename, target_resolution=(360, None), resize_algorithm='bilinear')
	# clip = VideoFileClip(filename, target_resolution=(299,299), resize_algorithm='bilinear')

	nr_frames = 0
	all_frames = None
	nr_frames = 0
	all_frames = None

	print('new video: ' + filename)
	fps = (32 / clip.duration) 

	for frames in clip.iter_frames(fps=fps):
		frame = np.expand_dims(frames, axis=0)
		if nr_frames == 0:
			all_frames = frame
		else:
			all_frames = np.append(all_frames, frame, axis=0)
		nr_frames += 1
	if nr_frames != 32:
		print("Alt numar de frameuri: " + str(nr_frames))
	return all_frames, nr_frames

# def read_data_kinetics(files, fps, old_videos={}):
# 	batch_videos = {}
# 	for i,filename in enumerate(files):
# 		try:
# 			video_name = filename.split('/')[-1].split('.')[0][:-14]
# 			if video_name.encode() not in old_videos:
# 				video, num_frames = read_video(filename, fps)
# 				batch_videos[video_name] = video
# 			else:
# 				print (video_name + "is already in dataset")
# 		except Exception as ex:
# 			print (ex)
# 			print("!!!! Problema cu fps !!!")
# 	return batch_videos

# read data => (train, test)
def read_data_mnist_artif(num_classes):
	num_elem = 6400
	num_frames = 10
	train_elem = 5000
	test_elem = 1400
	x = np.random.randint(255, size=(num_elem, num_frames, 64, 64, 3)).astype(float)
	y = np.random.randint(num_classes, size=(num_elem))
	y = np.eye(num_classes)[y]

	return [x[:train_elem], y[:train_elem]],[x[train_elem:], y[train_elem:]]

# read data when digit bounding boxes are known apriori
# read one pickle
def read_data_mnist_boxes(file, num_classes):
	num_elem = 1000
	num_frames = 10
	
	with open(file, 'rb') as fo:
		videos_dict = pickle.load(fo)
		x = videos_dict['videos']
		x = np.expand_dims(x, 4)
		# input is in [-0.5, 0.5]
		#x = x.astype(np.float32) / 255.0 - 0.5
		y = videos_dict['labels'].astype(int).squeeze()
		#y = np.ones(y.shape).astype(int)
		y = np.clip(y,0,num_classes-1)
		y = np.eye(num_classes)[y]

		digits = videos_dict['videos_digits']
		digits_coord = videos_dict['videos_digits_coords']
		digits_coord = (np.array(digits_coord)* 64 / 100).astype(int)
	x_dict = {}
	x_dict['videos'] = x
	x_dict['digits'] = digits
	x_dict['digits_coord'] = digits_coord
	return x_dict,y

# read data when digit bounding boxes are known apriori
# read a list of pickles
def read_data_mnist_boxes_files(files, num_classes):
	num_elem = 1000
	num_frames = 10
	all_x = None
	all_y = None
	for i,file in enumerate(files):
		with open(file, 'rb') as fo:
			videos_dict = pickle.load(fo)
			x = videos_dict['videos']
			x = np.expand_dims(x, 4)
			#x = x.astype(np.float32) / 255.0 - 0.5

			y = videos_dict['labels'].astype(int).squeeze()
			#y = np.ones(y.shape).astype(int)
			y = np.clip(y,0,num_classes-1)
			y = np.eye(num_classes)[y]

			digits = videos_dict['videos_digits']
			digits_coord = videos_dict['videos_digits_coords']

			x_dict = {}
			x_dict['videos'] = x
			x_dict['digits'] = digits
			x_dict['digits_coord'] = digits_coord

			if i == 0:
				all_x = x_dict
				all_y = y
			else:
				all_x = np.concatenate((all_x, x_dict), axis=0)
				all_y = np.concatenate((all_y, y_dict), axis=0)

	pdb.set_trace()
	return all_x,all_y

def read_data_mnist(file, num_classes = 65):
	num_elem = 1000
	num_frames = 10
	

	with open(file, 'rb') as fo:
		videos_dict = pickle.load(fo)
		x = videos_dict['videos']
		x = np.expand_dims(x, 4)
		# input is in [-0.5, 0.5]
		#x = x.astype(np.float32) / 255.0 - 0.5
		y = videos_dict['labels'].astype(int).squeeze()
		#y = np.ones(y.shape).astype(int)
		y = np.clip(y,0,num_classes-1)
		y = np.eye(num_classes)[y]
	return x,y

def read_data_mnist_files(files, num_classes = 65):
	num_elem = 1000
	num_frames = 10
	all_x = None
	all_y = None
	for i,file in enumerate(files):
		with open(file, 'rb') as fo:
			videos_dict = pickle.load(fo)
			x = videos_dict['videos']
			x = np.expand_dims(x, 4)
			#x = x.astype(np.float32) / 255.0 - 0.5

			y = videos_dict['labels'].astype(int).squeeze()
			#y = np.ones(y.shape).astype(int)
			y = np.clip(y,0,num_classes-1)
			y = np.eye(num_classes)[y]

			if i == 0:
				all_x = x
				all_y = y
			else:
				all_x = np.concatenate((all_x, x), axis=0)
				all_y = np.concatenate((all_y, y), axis=0)

	return all_x,all_y

def extract_from_record(features):
	pdb.set_trace()
	num_classes = 10
	num_frames = 10
	num_nodes = 9+4+1
	dim_feats = 512

	node_feats = tf.sparse_tensor_to_dense(features['node_feats'], default_value=0)
	target = tf.sparse_tensor_to_dense(features['target'], default_value=0)
	adj_matrix = tf.sparse_tensor_to_dense(features['adj_matrix'], default_value=0)
	# num_frames = tf.cast(features['num_frames'],tf.int32)
	# num_nodes = tf.cast(features['num_nodes'],tf.int32)

	# batch_size x num_frames x num_patches x 512
	node_feats_reshaped = tf.reshape(node_feats, [num_frames,num_nodes,dim_feats])
	target_reshaped = tf.reshape(target, [num_classes])
	adj_matrix_reshaped = tf.reshape(adj_matrix, [num_nodes, num_nodes]) 
	
	pdb.set_trace()
	return [node_feats_reshaped, target_reshaped, adj_matrix_reshaped, num_nodes, num_frames]

def read_tfrecord_mnist(tf_dir, num_epochs, batch_size):
	filenames = glob(tf_dir)
	feature = {'video_id':  tf.VarLenFeature(tf.float32),
				'video_feats': tf.VarLenFeature(tf.float32),
				'label': tf.VarLenFeature(tf.float32),
				'num_frames': tf.FixedLenFeature([], tf.int64) 
				}
	filename_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs, shuffle=True)

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(serialized_example, features=feature)

	node_feats, target, adj_matrix, \
		num_nodes,  nr_frames = tf.train.shuffle_batch(
											extract_from_record(features),
											batch_size=batch_size, 
											capacity=5000, 
											num_threads=1, 
											min_after_dequeue=0,
											allow_smaller_final_batch=True)

	return [node_feats, target, adj_matrix, num_nodes,  nr_frames]
	




##########################  DATASET TFRECORD   #####################

def read_video1(clip, fps):
	nr_frames = 0
	all_frames = None

	for frames in clip.iter_frames(fps=fps):
		frame = np.expand_dims(frames, axis=0)
		if nr_frames == 0:
			all_frames = frame
		else:
			all_frames = np.append(all_frames, frame, axis=0)
		nr_frames += 1
	
	return all_frames, nr_frames

def restore_inception_model(video_pl, ckpt_path, sess):
	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())	

	with  slim.arg_scope(inception.inception_v3_arg_scope()):
		logits, end_points = inception.inception_v3(inputs=video_pl, 
										num_classes=1001, 
										is_training=False,
										spatial_squeeze=False)
	saver = tf.train.Saver() 
	saver.restore(sess, ckpt_path)
	return logits, end_points

def restore_mnist_model( sess, dim_w, dim_h, hid_units=3, num_filters=512,ckpt_path=''):
	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer()) 

	all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	conv_vars = [var for var in all_variables if 'conv' in var.name and 'Adam' not in var.name]
	
	# checkpoint = tf.train.latest_checkpoint(ckpt_path)
	# saver = tf.train.Saver(conv_vars)
	# saver.restore(sess,  checkpoint)


def get_mnist_logits(video_item, video_pl, sess, end_points):
	video_shape = video_item.shape
	mnist_features = [[] for i in range(video_shape[0])]

	for i,frame in enumerate(video_item):
			frame = np.expand_dims(frame, axis=0)
			feed_dict = {video_pl: frame}
			predictions_val = sess.run(end_points['conv2'],feed_dict = feed_dict)	
			mnist_features[i] = predictions_val
	return np.asarray(mnist_features)

def get_mnist_logits_tensor(video_item, video_pl, sess, end_points):
	video_shape = video_item.shape
	mnist_features = [[] for i in range(video_shape[0])]

	for i,frame in enumerate(video_item):
		frame = np.expand_dims(frame, axis=0)
		feats = tf.reduce_mean(end_points['conv2'], axis=[1,2,3])	
			
	return feats


