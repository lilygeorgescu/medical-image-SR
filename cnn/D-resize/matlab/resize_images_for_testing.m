% folder_name = 'data/validation';
% folder_name = '../data/test';
folder_name = '../data/train';
files = dir(folder_name);
files(1:2) = [];  
resize_factor = 2;
input_folder_name = 'input_';  
for file_id = 1:numel(files)
   images_name = dir(strcat(folder_name, '/', files(file_id).name, '/transposed/'));
   images_name(1:2) = []; % delete . and ..
   folder_in = strcat(folder_name, '/', files(file_id).name, '/', input_folder_name);  
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
       image_name = strcat(folder_name, '/', files(file_id).name, '/transposed/', images_name(image_id).name); 
       image = imread(image_name); 
       [lines, cols] = size(image);
       in_image = imresize(image, [lines, round(cols/resize_factor)]);
       imwrite(in_image, strcat(folder_in, '/', images_name(image_id).name));
   end
end