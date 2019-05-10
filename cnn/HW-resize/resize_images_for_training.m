folder_name = 'data/train';
files = dir(folder_name);
files(1:2) = []; % delete . and .. 
dim_patch = 10;
stride = 9;
resize_factor = 2;
input_folder_name = sprintf('input_%d', dim_patch);
gt_folder_name =  sprintf('gt_%d', dim_patch);

for file_id = 1:numel(files)
   images_name = dir(strcat(folder_name, '/', files(file_id).name));
   images_name(1:2) = []; % delete . and ..
   folder_in = strcat(folder_name, '/', files(file_id).name, '/', input_folder_name);
   folder_gt = strcat(folder_name, '/', files(file_id).name, '/', gt_folder_name);
   
   if ~exist(folder_in, 'dir')
       mkdir(folder_in)
   else
       rmdir(folder_in, 's')
       mkdir(folder_in)
   end
   
   if ~exist(folder_gt, 'dir')
       mkdir(folder_gt)
   else
       rmdir(folder_gt, 's')
       mkdir(folder_gt)
   end
   
   idx_image = 0;
   for image_id = 1:numel(images_name)
       sprintf('%d/%d',file_id, image_id)
       if(images_name(image_id).isdir == 1)
           continue
       end
       image_name = strcat(folder_name, '/', files(file_id).name, '/', images_name(image_id).name); 
       image = imread(image_name); 
%        image = rgb2gray(image);
%        imwrite(image, image_name)
       idx_image = extract_patch_save_images(image, dim_patch, stride, resize_factor, folder_in, folder_gt, idx_image);
   end
end