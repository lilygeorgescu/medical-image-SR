function [] = create_clean_folder(folder_name)

       if ~exist(folder_name, 'dir')
           mkdir(folder_name)
       else
           rmdir(folder_name, 's')
           mkdir(folder_name)
       end
end

