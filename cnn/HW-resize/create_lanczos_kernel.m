mean_pixel = load('mean.txt') * 255 ;
a = 3;
value = (a * sin(mean_pixel * pi) * sin(mean_pixel * pi / a)) / (pi^2 * mean_pixel^2);

kernel = ones(a, a) * value;

fileID = fopen('lanczos_kernel.txt','w');
nbytes = fprintf(fileID, '%f', value);
fclose(fileID);

