% 设置当前目录
folder_path = 'F:\new\analysis\tu_p-v-a\v';

% 设置输出目录
output_folder_path = 'F:\new\analysis\tj\v';

% 如果输出目录不存在，则创建它
if ~exist(output_folder_path, 'dir')
    mkdir(output_folder_path);
end

% 获取目录中的所有tif文件
tif_files = dir(fullfile(folder_path, '*.tif'));

% 选择基准文件
base_file_path = fullfile(folder_path, "V-2021.tif"); % 请替换'基准文件名.tif'为实际的基准文件名

% 读取基准文件
[I_base, R_base] = geotiffread(base_file_path);

% 获取基准文件的行数和列数
[rows_base, cols_base] = size(I_base);

% 创建一个坐标网格
[col_base, row_base] = meshgrid(1:cols_base, 1:rows_base);

% 根据地理参考系统的特性选择计算经纬度坐标的方法
try
    % 尝试使用 intrinsicToGeographic 函数
    [lat_base, lon_base] = pix2latlon(R_base, row_base, col_base);
catch
    % 如果 intrinsicToGeographic 函数失败，则使用 R.XWorldLimits 和 R.YWorldLimits 属性
    lon_base = R_base.XWorldLimits(1) + (col_base-1).*R_base.CellExtentInWorldX;
    lat_base = R_base.YWorldLimits(2) - (row_base-1).*R_base.CellExtentInWorldY;
end

% 创建一个空的矩阵用于存储基准文件的数据
data_base = zeros(rows_base*cols_base, 3);

% 将经纬度坐标和像素值写入矩阵
for i = 1:rows_base
    for j = 1:cols_base
        data_base((i-1)*cols_base+j, :) = [lon_base(i,j), lat_base(i,j), I_base(i,j)];
    end
end

% 记录需要删除的行号
delete_rows = find(isnan(data_base(:,3)) | data_base(:,3) < 0 | data_base(:,3) > 1);

% 遍历所有tif文件，记录包含NaN值的行号
for k = 1:length(tif_files)
    % 获取tif文件的完整路径
    file_path = fullfile(folder_path, tif_files(k).name);
    
    % 读取tif文件
    [I,R] = geotiffread(file_path);

    % 获取数据的行数和列数
    [rows, cols] = size(I);

    % 创建一个坐标网格
    [col, row] = meshgrid(1:cols, 1:rows);

    % 根据地理参考系统的特性选择计算经纬度坐标的方法
    try
        % 尝试使用 intrinsicToGeographic 函数
        [lat, lon] = pix2latlon(R, row, col);
    catch
        % 如果 intrinsicToGeographic 函数失败，则使用 R.XWorldLimits 和 R.YWorldLimits 属性
        lon = R.XWorldLimits(1) + (col-1).*R.CellExtentInWorldX;
        lat = R.YWorldLimits(2) - (row-1).*R.CellExtentInWorldY;
    end

    % 创建一个空的矩阵用于存储数据
    data = zeros(rows*cols, 3);

    % 将经纬度坐标和像素值写入矩阵
    for i = 1:rows
        for j = 1:cols
            data((i-1)*cols+j, :) = [lon(i,j), lat(i,j), I(i,j)];
        end
    end

    % 记录包含NaN值的行号
    delete_rows = union(delete_rows, find(isnan(data(:,3)) | data(:,3) < 0 | data(:,3) > 1));
end

% 删除基准文件中记录的行号
data_base(delete_rows, :) = [];

% 初始化结果矩阵，包含经纬度和基准文件数据
result_data = data_base;

% 初始化表头
header = {'Latitude', 'Longitude', 'BaseFile'};

% 遍历所有tif文件
for k = 1:length(tif_files)
    % 获取tif文件的完整路径
    file_path = fullfile(folder_path, tif_files(k).name);
    
    % 读取tif文件
    [I,R] = geotiffread(file_path);

    % 获取数据的行数和列数
    [rows, cols] = size(I);

    % 创建一个坐标网格
    [col, row] = meshgrid(1:cols, 1:rows);

    % 根据地理参考系统的特性选择计算经纬度坐标的方法
    try
        % 尝试使用 intrinsicToGeographic 函数
        [lat, lon] = pix2latlon(R, row, col);
    catch
        % 如果 intrinsicToGeographic 函数失败，则使用 R.XWorldLimits 和 R.YWorldLimits 属性
        lon = R.XWorldLimits(1) + (col-1).*R.CellExtentInWorldX;
        lat = R.YWorldLimits(2) - (row-1).*R.CellExtentInWorldY;
    end

    % 创建一个空的矩阵用于存储数据
    data = zeros(rows*cols, 3);

    % 将经纬度坐标和像素值写入矩阵
    for i = 1:rows
        for j = 1:cols
            data((i-1)*cols+j, :) = [lon(i,j), lat(i,j), I(i,j)];
        end
    end

    % 删除记录的行号
    data(delete_rows, :) = [];

    % 将当前文件的数据添加到结果矩阵中
    result_data = [result_data, data(:,3)];

    % 添加文件名到表头
    [~, name, ~] = fileparts(tif_files(k).name);
    header{end+1} = name;
end

% 将结果矩阵写入CSV文件
csv_file_path = fullfile(output_folder_path, 'result.csv');
writecell([header; num2cell(result_data)], csv_file_path);

disp('处理完成');
