import os
import rasterio
from rasterio.enums import Resampling

# 输入文件夹路径
input_folder = r"your_input_folder"

# 自动在输入文件夹下生成一个 resampled 子文件夹
output_folder = os.path.join(input_folder, "resampled")
os.makedirs(output_folder, exist_ok=True)

# 目标分辨率（单位与原始数据一致，比如米）
target_resolution = 30  # 例如 30m

# 选择重采样方法：nearest, bilinear, cubic
resample_method = "bilinear"

# 将字符串映射到 rasterio 的 Resampling 枚举
resample_dict = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic
}

if resample_method not in resample_dict:
    raise ValueError(f"不支持的重采样方法: {resample_method}")

for file in os.listdir(input_folder):
    if file.endswith(".tif"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        with rasterio.open(input_path) as src:
            # 计算新的 transform
            transform = src.transform
            new_transform = rasterio.Affine(
                target_resolution, transform.b, transform.c,
                transform.d, -target_resolution, transform.f
            )

            # 计算新的行列数
            new_width = int((src.bounds.right - src.bounds.left) / target_resolution)
            new_height = int((src.bounds.top - src.bounds.bottom) / target_resolution)

            # 更新 profile
            profile = src.profile
            profile.update({
                'transform': new_transform,
                'width': new_width,
                'height': new_height
            })

            # 重采样
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=resample_dict[resample_method]
            )

            # 写出结果
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)

print(f"批量重采样完成！结果已保存到: {output_folder}")
