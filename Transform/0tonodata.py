import os
from osgeo import gdal

def process_special_tifs(folder_path, nodata_value=0):
    """
    只处理文件夹下的 geo_temporalCoherence.tif 和 velocity.tif。
    处理后保存为新文件：
      - geo_temporalCoherence.tif → coh_文件夹名.tif
      - velocity.tif → vel_文件夹名.tif
    """

    if not os.path.exists(folder_path):
        print(f"错误：路径不存在 - {folder_path}")
        return

    folder_name = os.path.basename(os.path.normpath(folder_path))
    print(f"开始处理文件夹: {folder_path}")

    # 定义目标文件映射
    target_files = {
        "geo_temporalCoherence.tif": f"coh_{folder_name}.tif",
        "velocity.tif": f"vel_{folder_name}.tif"
    }

    for src_name, out_name in target_files.items():
        src_path = os.path.join(folder_path, src_name)
        if not os.path.exists(src_path):
            print(f"跳过：未找到 {src_name}")
            continue

        try:
            ds = gdal.Open(src_path, gdal.GA_ReadOnly)
            if ds is None:
                print(f"无法打开文件: {src_name}")
                continue

            driver = gdal.GetDriverByName("GTiff")
            out_path = os.path.join(folder_path, out_name)

            # 创建副本并设置 NoData
            out_ds = driver.CreateCopy(out_path, ds, strict=0)
            for i in range(1, out_ds.RasterCount + 1):
                band = out_ds.GetRasterBand(i)
                band.SetNoDataValue(nodata_value)

            out_ds.FlushCache()
            out_ds = None
            ds = None

            print(f"已处理: {src_name} → {out_name}")

        except Exception as e:
            print(f"处理文件 {src_name} 时出错: {e}")

    print("处理完成！")

# ==========================================
# 在这里修改你的文件夹路径
# ==========================================
if __name__ == "__main__":
    my_folder_path = r"E:\shanxi\22\22_11_121"
    process_special_tifs(my_folder_path, nodata_value=0)
