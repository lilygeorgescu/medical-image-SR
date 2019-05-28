function [idx_image] = extract_patch_save_images_2_scales(image, dim_patch, stride, resize_factor_1, resize_factor_2, folder_in_1, folder_in_2, folder_gt, idx_image)
    [h, w] = size(image);
    kernel = fspecial('gaussian', [3 3], 0.5); 
    
    
    for i=1:stride:h-dim_patch+1
        for j=1:stride:w-dim_patch+1
            idx_image = idx_image + 1;
            gt_patch = image(i:i+dim_patch-1, j:j+dim_patch-1);
            in_patch_2 = imresize(imfilter(gt_patch, kernel), 1/resize_factor_1);
            in_patch_1 = imresize(imfilter(gt_patch, kernel), 1/resize_factor_2);
            if(sum(gt_patch(:)) == 0)
                continue;
            end 
            
            imwrite(gt_patch, strcat(folder_gt, sprintf('/%d.png', idx_image)));
            imwrite(in_patch_2, strcat(folder_in_2, sprintf('/%d.png', idx_image)));
            imwrite(in_patch_1, strcat(folder_in_1, sprintf('/%d.png', idx_image)));
        end
    end
end

