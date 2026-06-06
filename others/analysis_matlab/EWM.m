% 读取TIF图像文件夹中的所有图像
imageFolder = 'F:\SJJ\2023';
imageFiles = dir(fullfile(imageFolder, '*.tif'));

% 初始化变量
numImages = length(imageFiles);
maxSize = [0, 0];

% 找到最大的图像尺寸
for i = 1:numImages
    img = imread(fullfile(imageFolder, imageFiles(i).name));
    maxSize = max(maxSize, size(img));
end

% 预分配内存
imageVectors = NaN(prod(maxSize), numImages);

% 将每个图像调整为最大尺寸并转换为一维向量
for i = 1:numImages
    img = imread(fullfile(imageFolder, imageFiles(i).name));
    paddedImg = NaN(maxSize); % 创建填充后的图像
    paddedImg(1:size(img, 1), 1:size(img, 2), :) = img; % 填充图像

    % 将0值替换为一个非常小的正数
    paddedImg(paddedImg == 0) = eps;

    % 只处理0-1之间的数据，超出范围的数据设置为NaN
    paddedImg(paddedImg < 0 | paddedImg > 1) = NaN;

    imgVector = paddedImg(:); % 将图像转换为一维向量
    imageVectors(:, i) = imgVector;
end

% 计算每个向量的熵，忽略NaN值
entropyValues = zeros(1, numImages);
for i = 1:numImages
    validData = imageVectors(:, i);
    validData = validData(~isnan(validData)); % 忽略NaN值
    if isempty(validData)
        entropyValues(i) = NaN; % 如果没有有效数据，设置熵为NaN
    else
        p = validData / (sum(validData) + eps); % 计算概率分布，添加小常数以避免除零
        entropyValues(i) = -sum(p .* log2(p + eps)); % 计算熵
    end
end

% 打印熵值以检查是否有NaN
disp('熵值:');
disp(entropyValues);

% 计算权重并归一化
validEntropyValues = entropyValues(~isnan(entropyValues));
weights = (1 - validEntropyValues) / sum(1 - validEntropyValues);
weights = weights / sum(weights); % 归一化权重

% 输出权重
disp('权重:');
disp(weights);
