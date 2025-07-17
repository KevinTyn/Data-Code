% 读取文件夹中的所有.tif图像
imageFiles = dir('F:\new\analysis\tu\*.tif');
numImages = length(imageFiles);

% 读取图像并进行预处理
images = cell(1, numImages);
for i = 1:numImages
    img = imread(fullfile(imageFiles(i).folder, imageFiles(i).name));
    
    % 将图像转换为灰度图像
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % 归一化图像到0-1范围
    img = double(img) / 255;
    
    images{i} = img;
end

% 获取图像尺寸
[rows, cols] = size(images{1});

% 初始化Hurst指数图像
hurstImage = NaN(rows, cols, 'single'); % 使用单精度浮点数，并初始化为NaN

% 对每个像素点计算Hurst指数
for r = 1:rows
    for c = 1:cols
        % 提取像素点的时间序列
        pixelSeries = zeros(1, numImages);
        for i = 1:numImages
            pixelSeries(i) = images{i}(r, c);
        end
        
        % 只计算0-1范围内的数据
        if all(pixelSeries >= 0 & pixelSeries <= 1)
            % 计算该像素点的Hurst指数
            hurstImage(r, c) = estimateHurstIndex(pixelSeries);
        end
    end
end

% 显示Hurst指数图像
imshow(hurstImage, []);
colorbar;
title('Hurst Index Image');

% 读取第一个图像的投影信息
info = geotiffinfo(fullfile(imageFiles(1).folder, imageFiles(1).name));

% 保存Hurst指数图像到本地，保留原始图像的投影信息
geotiffwrite('F:\new\analysis\hurst\1\hurst_index_image.tif', hurstImage, info.SpatialRef, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

function H = estimateHurstIndex(signal)
    % 确保信号非空且长度大于1
    if isempty(signal) || length(signal) < 2
        H = NaN;
        return;
    end
    
    % 计算信号的累积和
    Y = cumsum(signal - mean(signal));
    
    % 计算R/S统计量
    R = max(Y) - min(Y);
    S = std(signal);
    
    % 确保S不为零
    if S == 0
        H = NaN;
        return;
    end
    
    RS = R / S;
    
    % 计算Hurst指数
    N = length(signal);
    H = log(RS) / log(N);
    
    % 确保Hurst指数在合理范围内
    if H < 0 || H > 1
        H = NaN;
    end
end
