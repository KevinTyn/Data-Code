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
parfor r = 1:rows
    for c = 1:cols
        % 提取像素点的时间序列
        pixelSeries = zeros(1, numImages);
        for i = 1:numImages
            pixelSeries(i) = images{i}(r, c);
        end
        
        % 只计算0-1范围内的数据，且数据长度大于一定阈值
        if all(pixelSeries >= 0 & pixelSeries <= 1) && numImages > 10
            % 计算该像素点的Hurst指数（使用极差法）
            hurstImage(r, c) = estimateHurstIndexRangeMethod(pixelSeries);
        end
    end
end

% 处理NaN值：将NaN值替换为0
hurstImage(isnan(hurstImage)) = 0;  % 将 NaN 替换为 0

% 显示Hurst指数图像
imagesc(hurstImage);  % 使用imagesc显示图像，自动处理NaN值
colorbar;
title('Hurst Index Image');

% 读取第一个图像的投影信息
info = geotiffinfo(fullfile(imageFiles(1).folder, imageFiles(1).name));

% 保存Hurst指数图像到本地，保留原始图像的投影信息
geotiffwrite('F:\new\analysis\hurst\1\hurst_index_image.tif', hurstImage, info.SpatialRef, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

% 新的估算Hurst指数的方法：极差法（Range Method）
function H = estimateHurstIndexRangeMethod(signal)
    % 确保信号非空且长度大于1
    if isempty(signal) || length(signal) < 2
        H = NaN;
        return;
    end
    
    % 设置多尺度分析的最小和最大尺度
    minScale = 5;  % 最小尺度
    maxScale = floor(length(signal) / 2);  % 最大尺度，避免分段太小
    
    % 初始化变量
    rangeValues = zeros(1, maxScale - minScale + 1);  % 存储不同尺度的极差
    
    % 对不同尺度进行分析
    for scale = minScale:maxScale
        % 将信号分为多个尺度段
        numSegments = floor(length(signal) / scale);
        segmentRanges = zeros(1, numSegments);
        
        for i = 1:numSegments
            % 提取当前尺度段
            segment = signal((i-1)*scale + 1:i*scale);
            
            % 计算该段的极差（最大值 - 最小值）
            segmentRanges(i) = max(segment) - min(segment);
        end
        
        % 计算该尺度的平均极差
        rangeValues(scale - minScale + 1) = mean(segmentRanges);
    end
    
    % 拟合尺度与极差之间的关系，斜率即为Hurst指数
    scales = minScale:maxScale;
    logScales = log(scales);
    logRangeValues = log(rangeValues);
    
    % 进行线性拟合，拟合斜率就是Hurst指数
    coeffs = polyfit(logScales, logRangeValues, 1);
    H = coeffs(1);  % 斜率即为Hurst指数
    
    % 确保Hurst指数在合理范围内
    if H < 0 || H > 1
        H = NaN;
    end
end
