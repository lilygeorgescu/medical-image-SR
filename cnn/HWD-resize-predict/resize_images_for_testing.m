
folder_name = 'data/test';
folder_output = 'data/test/';
files = dir(folder_name);
files(1:2) = [];
resize_factor = 2;
input_folder_name = 'input';

for file_id = 1:numel(files)
    images_name = dir(strcat(folder_name, '/', files(file_id).name, '/original/'));
    images_name(1:2) = []; % delete . and ..
    folder_in = strcat(folder_output, '/', files(file_id).name, '/', input_folder_name);
    
    if ~exist(folder_in, 'dir')
        mkdir(folder_in)
    else
        rmdir(folder_in, 's')
        mkdir(folder_in)
    end
    
    clear resize_d_images images
    for image_id = 1:numel(images_name)
        if(images_name(image_id).isdir == 1)
            continue
        end
        image_name = strcat(folder_name, '/', files(file_id).name, '/original/', images_name(image_id).name);
        image = imread(image_name);
        images(image_id, :, :) = image;
    end
    clear resize_d_images
    [lines, cols] = size(image);
    for line = 1:lines
        for col = 1:cols
            row = images(:, line, col);
            row = imresize(row, [round(numel(row)/resize_factor), 1]);
            resize_d_images(:, line, col) = row;
        end
    end
    
    for image_idx = 1:size(resize_d_images, 1)
        image = squeeze(resize_d_images(image_idx,:,:));
        image = imresize(image, 1/resize_factor);
        imwrite(image, strcat(folder_in, '/', sprintf('%.4d', image_idx), '.png'));
    end 
end