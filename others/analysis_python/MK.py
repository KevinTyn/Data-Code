import numpy as np
import rasterio
from scipy.stats import norm

# 读取文件夹中的TIFF文件并存储图像数据
folder_path = 'E:\\矿大\\呼包鄂榆\\数据集\\1\\analysis\\hurst\\yx'
file_list = dir(fullfile(folder_path, '*.tif'))

# 初始化cell数组来存储图像数据
images = []
for i in range(len(file_list)):
    file_name = os.path.join(folder_path, file_list[i].name)
    with rasterio.open(file_name) as src:
        images.append(src.read(1))
        if i == 0:
            profile = src.profile  # 获取空间参考信息

# 获取图像尺寸
rows, cols = images[0].shape

# 初始化结果矩阵
H = np.zeros((rows, cols), dtype=np.float32)
p_value = np.zeros((rows, cols), dtype=np.float32)
S = np.zeros((rows, cols), dtype=np.float32)
Z = np.zeros((rows, cols), dtype=np.float32)
trend_category = np.zeros((rows, cols), dtype=np.uint8)  # 用于存储趋势分类

# 对每个像素进行Mann-Kendall检验
for r in range(rows):
    for c in range(cols):
        pixel_values = np.array([images[i][r, c] for i in range(len(images))])

        # 进行MK检验
        H[r, c], p_value[r, c], S[r, c], Z[r, c] = mann_kendall(pixel_values, 0.05)

        # 根据Z值和显著性进行分类
        if Z[r, c] > 0 and H[r, c] == 1:
            trend_category[r, c] = 1  # 显著正趋势
        elif Z[r, c] > 0 and H[r, c] == 0:
            trend_category[r, c] = 2  # 不显著正趋势
        elif Z[r, c] == 0:
            trend_category[r, c] = 3  # 无显著变化
        elif Z[r, c] < 0 and H[r, c] == 0:
            trend_category[r, c] = 4  # 不显著负趋势
        elif Z[r, c] < 0 and H[r, c] == 1:
            trend_category[r, c] = 5  # 显著负趋势

# 保存结果为GeoTIFF
output_file = 'E:\\矿大\\呼包鄂榆\\数据集\\1\\analysis\\hurst\\yx\\MK_trend_category.tif'

# 更新输出文件的profile
profile.update(
    dtype=rasterio.uint8,  # 保存为8位无符号整数
    count=1,  # 单波段
    compress='lzw'  # 使用LZW压缩
)

# 保存分类结果为GeoTIFF
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(trend_category, 1)

print(f"趋势分类图已保存为: {output_file}")

# 显示结果
plt.imshow(trend_category, cmap='tab10', interpolation='none')  # 使用离散颜色图
plt.colorbar()
plt.title('Mann-Kendall Trend Categories')
plt.show()


# Mann-Kendall检验函数
def mann_kendall(V, alpha):
    n = len(V)
    S = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            S += np.sign(V[j] - V[k])

    # 计算S的方差
    var_S = (n * (n - 1) * (2 * n + 5)) / 18

    # 计算检验统计量Z
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S == 0:
        Z = 0
    else:
        Z = (S + 1) / np.sqrt(var_S)

    # 计算p-value
    p_value = 2 * (1 - norm.cdf(abs(Z)))

    # 判断是否拒绝零假设
    H = abs(Z) > norm.ppf(1 - alpha / 2)

    return H, p_value, S, Z

