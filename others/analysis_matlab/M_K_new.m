% 设置文件夹路径
folder_path = 'F:\new\analysis\tu';

% 获取文件夹中所有TIFF文件的列表
file_list = dir(fullfile(folder_path, '*.tif'));

% 读取第一个图像的投影信息
info = geotiffinfo(fullfile(folder_path, file_list(1).name));

% 读取第一张图像以获取图像尺寸
sample_image = imread(fullfile(folder_path, file_list(1).name));
[rows, cols] = size(sample_image);

% 初始化三维数组来存储所有图像数据
num_images = length(file_list);
images = zeros(rows, cols, num_images);

% 读取所有TIFF文件并存储图像数据
for i = 1:num_images
    file_name = fullfile(folder_path, file_list(i).name);
    images(:,:,i) = imread(file_name);
end

% 初始化结果矩阵
H = zeros(rows, cols);
p_value = zeros(rows, cols);
S = zeros(rows, cols);
Z = zeros(rows, cols);

% 对每个像素进行MK检验
for r = 1:rows
    for c = 1:cols
        % 提取每个像素的时间序列
        pixel_values = squeeze(images(r, c, :));
        
        % 进行MK检验
        [H(r, c), p_value(r, c), S(r, c), Z(r, c)] = mann_kendall(pixel_values, 0.1);
    end
end

% 显示结果
figure;
subplot(2, 2, 1);
imagesc(H);
colorbar;
title('Mann-Kendall Test H');

subplot(2, 2, 2);
imagesc(p_value);
colorbar;
title('p-value');

subplot(2, 2, 3);
imagesc(S);
colorbar;
title('S value');

subplot(2, 2, 4);
imagesc(Z);
colorbar;
title('Z value');

% 保存结果为TIFF文件
output_folder = 'F:\new\analysis\mk\2';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 保存Mann-Kendall检验结果图像到本地，保留原始图像的投影信息
geotiffwrite(fullfile(output_folder, 'MK_H_image.tif'), H, info.SpatialRef, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

geotiffwrite(fullfile(output_folder, 'MK_p_value_image.tif'), p_value, info.SpatialRef, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

geotiffwrite(fullfile(output_folder, 'MK_S_value_image.tif'), S, info.SpatialRef, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

geotiffwrite(fullfile(output_folder, 'MK_Z_value_image.tif'), Z, info.SpatialRef, ...
    'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);


function [H, p_value, S, Z] = mann_kendall(V, alpha)
    % Mann-Kendall trend test
    % V: time series data
    % alpha: significance level (e.g., 0.05)
    
    n = length(V);
    S = 0;
    for k = 1:n-1
        for j = k+1:n
            S = S + sign(V(j) - V(k));
        end
    end
    
    % 计算S的方差
    var_S = (n*(n-1)*(2*n+5))/18;
    
    % 计算检验统计量Z
    if S > 0
        Z = (S - 1) / sqrt(var_S);
    elseif S == 0
        Z = 0;
    else
        Z = (S + 1) / sqrt(var_S);
    end
    
    % 计算p值
    p_value = 2 * (1 - normcdf(abs(Z), 0, 1));
    
    % 判断是否拒绝原假设
    H = abs(Z) > norminv(1 - alpha/2);
end
