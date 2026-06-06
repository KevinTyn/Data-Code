import os
import rasterio

# 你的 txt 文件所在目录
folder = r"E:\fhly\Daily_Snow_depth_dataset(1985_2024)\1"

# 遍历目录下所有 txt 文件
for txtname in os.listdir(folder):
    if txtname.lower().endswith(".txt"):
        in_file = os.path.join(folder, txtname)
        out_file = os.path.join(folder, os.path.splitext(txtname)[0] + ".tif")

        try:
            # 打开 ASCII 栅格（txt）
            with rasterio.open(in_file) as src:
                profile = src.profile
                data = src.read(1)

            # 更新驱动为 GeoTIFF
            profile.update(driver="GTiff")

            # 保存为 tif，路径与 txt 相同
            with rasterio.open(out_file, "w", **profile) as dst:
                dst.write(data, 1)

            print(f"已完成: {txtname} -> {out_file}")
        except Exception as e:
            print(f"转换失败: {txtname}, 错误信息: {e}")

print("所有 txt 已转换为 tif，并保存在原目录下！")
