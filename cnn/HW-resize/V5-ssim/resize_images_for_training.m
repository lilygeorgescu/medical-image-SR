folder_name = 'data/train';
files = dir(folder_name);
files(1:2) = []; % delete . and .. 
dim_patch = 28;
stride = 14;
resize_factor_1 = 2;
resize_factor_2 = 4;
input_folder_name_1 = sprintf('input_%d_%d', dim_patch, resize_factor_1);
input_folder_name_2 = sprintf('input_%d_%d', dim_patch, resize_factor_2);
gt_folder_name =  sprintf('gt_%d_%d_%d', dim_patch, resize_factor_1, resize_factor_2); 

for file_id = 1:numel(files)
   images_name = dir(strcat(folder_name, '/', files(file_id).name));
   images_name(1:2) = []; % delete . and ..
   
   folder_in_1 = strcat(folder_name, '/', files(file_id).name, '/', input_folder_name_1);  
   folder_in_2 = strcat(folder_name, '/', files(file_id).name, '/', input_folder_name_2);
   
   folder_gt = strcat(folder_name, '/', files(file_id).name, '/', gt_folder_name);
   
   create_clean_folder(folder_in_1)
   create_clean_folder(folder_in_2)
   create_clean_folder(folder_gt)
   
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
       idx_image = extract_patch_save_images_2_scales(image, dim_patch, stride, resize_factor_1, resize_factor_2, ...
       folder_in_1, folder_in_2, folder_gt, idx_image);
   end
end