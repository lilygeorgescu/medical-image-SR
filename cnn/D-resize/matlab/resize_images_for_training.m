folder_name = '../data/train';
base_folder_in = '/original/';
files = dir(folder_name);
files(1:2) = []; % delete . and .. 
% min dim 27 => max dim patch 26 -> 20 -> 14
dim_patch_w = 14;
dim_patch_h = 14;
stride = 10;
resize_factor = 2;
input_folder_name = sprintf('input_orig_%d_%d', dim_patch_w, dim_patch_h);
gt_folder_name =  sprintf('gt_orig_%d_%d', dim_patch_w, dim_patch_h);

for file_id = 1:numel(files)
   images_name = dir(strcat(folder_name, '/', files(file_id).name, base_folder_in));
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
       image_name = strcat(folder_name, '/', files(file_id).name, base_folder_in, images_name(image_id).name); 
       image = imread(image_name);  
       
       idx_image = extract_patch_save_images(image, dim_patch_w, dim_patch_h, stride, resize_factor, folder_in, folder_gt, idx_image);
   end
end