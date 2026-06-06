import os
import glob
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# ====== 你只需要改这两个路径 ======
shp_path = r"F:\Data\sar_prep\SHP\shanxi_1\shanxi_1.shp"
# shp_path = r"F:\Data\sar_prep\SHP\fhly\fhly.shp"# 矢量文件路径
tif_folder = r"E:\fhly\Annual_Land_cover_CLDC30m(1990_2024)"               # 存放tif文件的文件夹
# =================================

# 在tif文件夹下新建一个子文件夹 clipped
output_folder = os.path.join(tif_folder, "clipped_shanxi")
os.makedirs(output_folder, exist_ok=True)

# 读取矢量文件
gdf = gpd.read_file(shp_path)
geoms = gdf.geometry.values

# 遍历所有tif文件
for tif_path in glob.glob(os.path.join(tif_folder, "*.tif")):
    with rasterio.open(tif_path) as src:
        # 裁剪
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

        # 更新元数据
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # 输出文件名
        fname = os.path.basename(tif_path)
        out_path = os.path.join(output_folder, f"clipped_{fname}")

        # 保存结果
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)

print(f"批量裁剪完成！结果保存在: {output_folder}")
