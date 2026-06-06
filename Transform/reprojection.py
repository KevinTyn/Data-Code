import os
import csv
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ==========================================
#              全局参数设置区
# ==========================================
input_folder = r"E:\data_shanxi_fhly\Annual_Nighttime_light_500m(2013_2024)\clipped_shanxi"

TARGET_EPSG = 32649  # 目标 EPSG 代码 (例如: 32649)
TARGET_RESOLUTION = 100  # 目标栅格大小 (单位: 米)

# 重采样方法选择 (请填写下方选项对应的数字 1-4):
# 1 = 最邻近插值 (Nearest)  [推荐用于：土地利用、土壤类型等分类数据]
# 2 = 双线性插值 (Bilinear) [推荐用于：DEM高程、温度等连续数据]
# 3 = 三次卷积插值 (Cubic)  [推荐用于：需要平滑边缘的连续数据]
# 4 = 众数插值 (Mode)       [推荐用于：分类数据，提取窗口内的主导类别]
TARGET_RESAMPLING_METHOD = 2

# ==========================================

# 建立重采样映射字典
resampling_map = {
    1: ("最邻近插值 (Nearest)", Resampling.nearest),
    2: ("双线性插值 (Bilinear)", Resampling.bilinear),
    3: ("三次卷积插值 (Cubic)", Resampling.cubic),
    4: ("众数插值 (Mode)", Resampling.mode)
}

# 获取当前选择的插值方法（如果填错，默认回退到最邻近插值）
selected_name, selected_method = resampling_map.get(
    TARGET_RESAMPLING_METHOD,
    ("最邻近插值 (Nearest)", Resampling.nearest)
)

output_folder = os.path.join(input_folder, f"UTM")
os.makedirs(output_folder, exist_ok=True)

results = []
non_target_files = []

print(f"--- 任务配置信息 ---")
print(f"目标坐标系: EPSG:{TARGET_EPSG}")
print(f"目标分辨率: {TARGET_RESOLUTION}m")
print(f"重采样方法: {selected_name}")
print(f"--------------------\n")

# 1. 遍历文件并检查状态
print("正在扫描文件...")
for file in os.listdir(input_folder):
    if file.lower().endswith(".tif"):
        file_path = os.path.join(input_folder, file)
        with rasterio.open(file_path) as src:
            crs = src.crs
            epsg = crs.to_epsg() if crs else None

            results.append({
                "文件名": file,
                "原始坐标系": str(crs) if crs else "无",
                "原始EPSG": epsg if epsg else "未知",
                "最终EPSG": epsg if epsg else "未知",
                "是否已重投影": "否",
                "重采样方法": "无",
                "输出路径": ""
            })

            if epsg != TARGET_EPSG:
                non_target_files.append(file_path)

# 2. 批量重投影与重采样
if non_target_files:
    print(f"\n检测到 {len(non_target_files)} 个非 EPSG:{TARGET_EPSG} 的文件，开始处理...")
    dst_crs = f"EPSG:{TARGET_EPSG}"

    for file_path in non_target_files:
        file = os.path.basename(file_path)
        with rasterio.open(file_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=(TARGET_RESOLUTION, TARGET_RESOLUTION)
            )

            kwargs = src.meta.copy()
            kwargs.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            output_path = os.path.join(output_folder,
                                       f"{os.path.splitext(file)[0]}_epsg{TARGET_EPSG}_{TARGET_RESOLUTION}m.tif")
            with rasterio.open(output_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=selected_method
                    )

                try:
                    cmap = src.colormap(1)
                    if cmap:
                        dst.write_colormap(1, cmap)
                except Exception:
                    pass

        # 更新结果表
        for r in results:
            if r["文件名"] == file:
                r["是否已重投影"] = "是"
                r["最终EPSG"] = TARGET_EPSG
                r["重采样方法"] = selected_name
                r["输出路径"] = output_path
        print(f"✅ 已保存: {os.path.basename(output_path)}")
else:
    print(f"所有文件均已是 EPSG:{TARGET_EPSG}，无需处理。")

# 3. 导出处理台账
csv_path = os.path.join(output_folder, "坐标系汇总表.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["文件名", "原始坐标系", "原始EPSG", "最终EPSG", "是否已重投影", "重采样方法",
                                           "输出路径"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n🎉 处理完成！汇总表已保存到: {csv_path}")