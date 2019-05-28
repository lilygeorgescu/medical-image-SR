 
folder_images = 'D:/disertatie/code/read-nii/image/'; 
folder_in = 'D:/disertatie/code/read-nii/image/input_27';
resize_factor = 2;
 
images_name = dir(folder_images);
images_name(1:2) = []; % delete . and ..

if ~exist(folder_in, 'dir')
   mkdir(folder_in)
else
   rmdir(folder_in, 's')
   mkdir(folder_in)
end
for image_id = 1:numel(images_name)
   if(images_name(image_id).isdir == 1)
       continue
   end
   image_name = strcat(folder_images, '/', images_name(image_id).name); 
   image = imread(image_name); 
   in_image = imresize(image, 1/resize_factor);
   imwrite(in_image, strcat(folder_in, '/', images_name(image_id).name));
end 