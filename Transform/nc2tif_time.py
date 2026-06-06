import os
import xarray as xr
import rasterio
from rasterio.transform import from_origin

# 指定存放 nc 文件的文件夹路径
input_folder = r"E:\fhly\Annual_Mean_temperature_1km(1985_2023)\tmp_2024"   # ← 修改为你的 nc 文件夹路径

# 遍历文件夹下所有 nc 文件
for file in os.listdir(input_folder):
    if file.endswith(".nc"):
        nc_path = os.path.join(input_folder, file)
        print(f"正在处理: {nc_path}")

        # 打开 nc 文件
        ds = xr.open_dataset(nc_path)

        # 选择变量（默认取第一个变量，你也可以改成具体变量名）
        var_name = list(ds.data_vars)[0]
        data = ds[var_name]

        # 获取经纬度
        lon = ds['lon'].values
        lat = ds['lat'].values

        # 构建仿射变换
        transform = from_origin(lon.min(), lat.max(), lon[1]-lon[0], lat[0]-lat[1])

        # 创建同名文件夹（去掉扩展名）
        nc_name = os.path.splitext(file)[0]   # 例如 pet_2020
        tif_folder = os.path.join(input_folder, nc_name)
        os.makedirs(tif_folder, exist_ok=True)

        # 从文件名中提取年份（假设格式固定为 pet_YYYY.nc）
        year = nc_name.split("_")[-1]

        # 遍历 time 维度，每个时间片保存成一个单独的 tif
        for i in range(data.sizes['time']):
            band_data = data.isel(time=i).values

            # 输出文件名：年份 + 月份
            tif_name = f"{year}_{i+1:02d}.tif"
            tif_path = os.path.join(tif_folder, tif_name)

            with rasterio.open(
                tif_path,
                'w',
                driver='GTiff',
                height=band_data.shape[0],
                width=band_data.shape[1],
                count=1,
                dtype=str(band_data.dtype),
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(band_data, 1)

            print(f"已保存: {tif_path}")

        ds.close()
