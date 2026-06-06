import rasterio
import pandas as pd
import numpy as np

# 输入 GeoTIFF 文件路径
tif_path = r"D:\BaiduSyncdisk\Code\pycharm\shanxi\mosaic\temp\overlap_filtered2.tif"
csv_path = r"D:\BaiduSyncdisk\Code\pycharm\shanxi\mosaic\temp\output2.csv"

with rasterio.open(tif_path) as src:
    data = src.read(1)  # 读取第一波段
    nodata = src.nodata
    transform = src.transform

    # 找到有效像元（非 nodata）
    mask = data != nodata
    rows, cols = np.where(mask)

    # 转换为经纬度坐标
    lons, lats = rasterio.transform.xy(transform, rows, cols)

    # 提取对应值
    values = data[rows, cols]

# 生成 DataFrame，前两列是经纬度
df = pd.DataFrame({
    "lon": lons,
    "lat": lats,
    "value": values
})

# 保存为 CSV
df.to_csv(csv_path, index=False)
