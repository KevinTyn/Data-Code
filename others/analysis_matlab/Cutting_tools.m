% 图像文件夹路径
folder = 'F:\TU\00\1';

% 获取文件夹中的所有图像文件
all_files = dir(fullfile(folder, '*.jpg'));

% 读取第一张图像
img = imread(fullfile(all_files(1).folder, all_files(1).name));

% 提示用户选择使用交互工具或手动输入裁剪区域
use_manual_input = input('Do you want to manually input the crop values? [1(yes)/2(no)]: ', 's');

if strcmpi(use_manual_input, '1')
    % 手动输入裁剪区域
    x = input('Enter the x coordinate of the top left corner: ');
    y = input('Enter the y coordinate of the top left corner: ');
    width = input('Enter the width of the crop area: ');
    height = input('Enter the height of the crop area: ');
    rect = [x, y, width, height];
    cropped_img = imcrop(img, rect);
else
    % 使用交互式工具选择裁剪区域
    [cropped_img, rect] = imcrop(img);
end

% 保存裁剪后的第一张图像
[~, name, ext] = fileparts(all_files(1).name);
imwrite(cropped_img, fullfile(folder, sprintf('%s_cropped%s', name, ext)));

% 打印裁剪区域的位置和大小
fprintf('The crop area for %s is [x, y, width, height] = [%f, %f, %f, %f]\n', all_files(1).name, rect);

% 遍历所有其他图像文件
for i = 2:length(all_files)
    % 读取图像
    img = imread(fullfile(all_files(i).folder, all_files(i).name));
    
    % 使用之前选择的裁剪区域裁剪图像
    cropped_img = imcrop(img, rect);
    
    % 保存裁剪后的图像
    [~, name, ext] = fileparts(all_files(i).name);
    imwrite(cropped_img, fullfile(folder, sprintf('%s_cropped%s', name, ext)));
end
