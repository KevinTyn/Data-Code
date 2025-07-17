% 设置母文件夹路径
parentFolderPath = 'F:\毕业论文\mrsei\相关性\sd';  % 替换为你的母文件夹路径

% 获取母文件夹下所有子文件夹
subFolders = dir(parentFolderPath);
subFolders = subFolders([subFolders.isdir] & ~startsWith({subFolders.name}, '.'));  % 仅选择子文件夹

% 遍历所有子文件夹
for k = 1:length(subFolders)
    % 获取当前子文件夹路径
    folderPath = fullfile(parentFolderPath, subFolders(k).name);
    
    % 获取当前文件夹下的所有.tif文件
    files = dir(fullfile(folderPath, '*.tif'));
    
    if isempty(files)
        continue;  % 如果当前文件夹下没有.tif文件，则跳过
    end

    % 初始化一个矩阵用于存储相关性值
    numFiles = length(files);
    correlationMatrix = zeros(numFiles, numFiles);

    % 读取每个图像并计算两两之间的相关性
    for i = 1:numFiles
        % 读取图像
        img1 = imread(fullfile(folderPath, files(i).name));
        img1 = double(img1);  % 转为double类型
        
        % 提取像素值在0到1之间的部分
        mask1 = (img1 >= 0 & img1 <= 1);  % 创建掩膜，选择0到1之间的像素
        img1_valid = img1(mask1);  % 提取有效像素
        
        for j = i:numFiles
            % 读取另一张图像
            img2 = imread(fullfile(folderPath, files(j).name));
            img2 = double(img2);  % 转为double类型
            
            % 提取像素值在0到1之间的部分
            mask2 = (img2 >= 0 & img2 <= 1);  % 创建掩膜，选择0到1之间的像素
            img2_valid = img2(mask2);  % 提取有效像素
            
            % 找到同行同列有效像素
            commonMask = mask1 & mask2;  % 取两幅图像在相同位置的有效像素
            img1_common = img1(commonMask);  % 提取同行同列有效像素的值
            img2_common = img2(commonMask);  % 提取同行同列有效像素的值
            
            % 计算有效像素部分的相关性系数 (皮尔逊相关系数)
            if length(img1_common) > 1  % 确保有效像素数量大于1
                corrValue = corr(img1_common(:), img2_common(:));  % 计算相关性系数
                corrValue = round(corrValue, 2);  % 保留两位小数
            else
                corrValue = NaN;  % 如果有效像素数量太少，则置为NaN
            end
            
            % 存储相关性值
            correlationMatrix(i, j) = corrValue;
            correlationMatrix(j, i) = corrValue;  % 相关性矩阵是对称的
        end
    end

    % 归一化相关性矩阵
    minCorr = min(correlationMatrix(:), [], 'omitnan');  % 找到最小值
    maxCorr = max(correlationMatrix(:), [], 'omitnan');  % 找到最大值

    % normalizedCorrelationMatrix = (correlationMatrix - minCorr) / (maxCorr - minCorr);

    normalizedCorrelationMatrix = correlationMatrix;


    % 确保所有相关性矩阵的值保留两位小数
    normalizedCorrelationMatrix = round(normalizedCorrelationMatrix, 2);

    % 创建包含文件名的矩阵
    fileNames = {files.name};  % 提取文件名

    % 将文件名添加到行和列
    normalizedCorrelationMatrixWithNames = array2table(normalizedCorrelationMatrix, ...
        'RowNames', fileNames, 'VariableNames', fileNames);

    % 输出带有文件名的归一化相关性矩阵
    disp(['带有文件名的归一化相关性矩阵（' subFolders(k).name '）：']);
    disp(normalizedCorrelationMatrixWithNames);

    % 生成保存路径，并保存为CSV文件
    outputFilePath = fullfile(folderPath, 'normalized_correlation_with_filenames.csv');
    writetable(normalizedCorrelationMatrixWithNames, outputFilePath, 'WriteRowNames', true);
    disp(['已保存结果到：' outputFilePath]);
end
