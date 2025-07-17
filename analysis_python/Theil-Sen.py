import os
import re
import numpy as np
import rasterio
from scipy.stats import theilslopes
import matplotlib.pyplot as plt

# 读取tif文件并提取数值
def read_tif(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        if np.isnan(data).any():
            print(f"NaN values detected in file {file_path}.")
            # 处理 NaN 值，例如用 0 或其他合适的值填充
            data = np.nan_to_num(data, nan=0)
        return data


# 文件夹路径
folder_path = r"F:\new\analysis\tu"

# 读取文件夹中的所有tif文件
tif_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.tif')]
years = []
values = []

# 从文件名中提取年份
for file in tif_files:
    print(f"Processing file: {file}")
    match = re.search(r'\d{4}', file)
    if match:
        year = int(match.group())
        data = read_tif(file)
        mean_value = np.mean(data)
        print(f"Year: {year}, Mean value: {mean_value}")
        years.append(year)
        values.append(mean_value)

years = np.array(years)
values = np.array(values)

# 检查years和values是否长度一致
if len(years) != len(values):
    raise ValueError(f"Incompatible lengths ! ({len(values)}<>{len(years)})")

print(f"Years: {years}")
print(f"Values: {values}")

# 计算Theil-Sen估计
slope, intercept, lower_slope, upper_slope = theilslopes(values, years, 0.95)


# 打印结果
print(f"斜率: {slope}")
print(f"截距: {intercept}")
print(f"95%置信区间的下限斜率: {lower_slope}")
print(f"95%置信区间的上限斜率: {upper_slope}")

# 绘制结果
plt.scatter(years, values, color='blue', label='数据点')
plt.plot(years, intercept + slope * years, 'r-', label='Theil-Sen估计')
plt.fill_between(years, intercept + lower_slope * years, intercept + upper_slope * years, color='pink', alpha=0.3, label='95%置信区间')
plt.xlabel('年份')
plt.ylabel('值')
plt.legend()
plt.show()
