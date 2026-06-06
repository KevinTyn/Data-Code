import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

# 打开 HDF5 文件
file_path = r"H:\23_113_116\mintpy\geo\geo_velocity.h5"
dataset_name = "velocity"

with h5py.File(file_path, "r") as f:
    data = f[dataset_name][:]   # 读取二维数据

# 如果没有地理坐标信息，就随便定义一个仿射变换
# (左上角x, 左上角y, 像元宽度, 像元高度)
transform = from_origin(0, 0, 1, 1)

# 保存为 GeoTIFF
out_tif = "output.tif"
with rasterio.open(
    out_tif,
    "w",
    driver="GTiff",
    height=data.shape[0],
    width=data.shape[1],
    count=1,
    dtype=data.dtype,
    crs="+proj=latlong",   # 如果没有坐标系，可以先随便写一个
    transform=transform,
) as dst:
    dst.write(data, 1)

print(f"已保存为 {out_tif}")
