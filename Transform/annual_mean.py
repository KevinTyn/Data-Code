import os
import glob
import numpy as np
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed

root_folder = r"C:\Users\Kevin\Desktop\中国逐月1km分辨率太阳辐射数据（2000-2024年）"
output_folder = os.path.join(root_folder, "annual_results")
os.makedirs(output_folder, exist_ok=True)

def process_tifs(tif_files, output_name):
    """处理一组 tif 文件，计算平均值并输出"""
    if not tif_files:
        return f"{output_name}: 没有tif文件，跳过"

    with rasterio.open(tif_files[0]) as src0:
        ref_shape = src0.shape
        ref_profile = src0.profile
        nodata_value = src0.nodata if src0.nodata is not None else -9999
        data_sum = np.zeros(ref_shape, dtype=np.float64)
        data_count = np.zeros(ref_shape, dtype=np.int32)

    for f in tif_files:
        with rasterio.open(f) as src:
            if src.shape != ref_shape:
                raise ValueError(f"{f} 的行列数与参考文件不一致")
            data = src.read(1).astype(np.float64)

            if src.nodata is not None:
                mask = (data != src.nodata)
            else:
                mask = np.ones_like(data, dtype=bool)

            data_sum[mask] += data[mask]
            data_count[mask] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        data_mean = np.where(data_count > 0, data_sum / data_count, nodata_value)

    ref_profile.update(dtype=rasterio.float32, count=1, nodata=nodata_value)

    output_file = os.path.join(output_folder, f"{output_name}_annual_mean.tif")
    with rasterio.open(output_file, "w", **ref_profile) as dst:
        dst.write(data_mean.astype(rasterio.float32), 1)

    return f"{output_name}: 已完成 -> {output_file}"

def process_year(year_folder):
    """处理单个年份文件夹"""
    year_path = os.path.join(root_folder, year_folder)
    tif_files = glob.glob(os.path.join(year_path, "*.tif"))
    return process_tifs(tif_files, year_folder)

if __name__ == "__main__":
    # 找到子文件夹
    year_folders = [
        f for f in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, f)) and f != "annual_results"
    ]

    if year_folders:
        # 有子文件夹 -> 并行处理每个年份
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_year, y): y for y in year_folders}
            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print(f"{futures[future]} 出错: {e}")
    else:
        # 没有子文件夹 -> 直接处理根目录下的所有 tif
        tif_files = glob.glob(os.path.join(root_folder, "*.tif"))
        result = process_tifs(tif_files, "all_years")
        print(result)
