import os
import numpy as np
import rasterio

# ================= 配置区域 =================
ROOT_DIR = r'E:\shanxi\21_23'  # 数据根目录
VEL_PREFIX = 'vel_'  # 速度文件前缀
IA_PREFIX = 'ia_'  # 入射角文件前缀 (匹配ID模式)
FIXED_IA_NAME = 'incidenceAngle.tif'  # 新增：固定的入射角文件名
SUFFIX = '.tif'  # 后缀名
OUT_SUFFIX = '_ia.tif'  # 输出文件的后缀


# ===========================================

def process_tif_pair(vel_path, ia_path, out_path):
    """
    读取速度和入射角文件，计算垂直形变并输出
    """
    print(f"正在处理: {os.path.basename(vel_path)}")
    print(f"  -> 使用入射角文件: {os.path.basename(ia_path)}")  # 打印确认用了哪个入射角文件

    try:
        # 1. 打开速度文件 (作为主文件，提供元数据)
        with rasterio.open(vel_path) as src_vel:
            vel_data = src_vel.read(1)  # 读取第一波段
            meta = src_vel.meta.copy()  # 复制元数据
            nodata_val = src_vel.nodata  # 获取无效值标记

            # 2. 打开入射角文件
            with rasterio.open(ia_path) as src_ia:
                # 检查尺寸是否匹配
                if src_vel.width != src_ia.width or src_vel.height != src_ia.height:
                    print(
                        f"  [Error] 尺寸不匹配 (Vel: {src_vel.width}x{src_vel.height}, IA: {src_ia.width}x{src_ia.height})，跳过！")
                    return

                ia_data = src_ia.read(1)

            # 3. 核心计算: V_vert = V_los / cos(theta)

            # 将入射角转为弧度
            ia_rad = np.deg2rad(ia_data)

            # 计算分母 cos(theta)
            cos_theta = np.cos(ia_rad)

            # --- 数据清洗与计算 ---
            # 假设 cos_theta < 0.1 是异常的 (入射角接近90度)
            valid_mask = (cos_theta > 0.1)

            # 初始化输出矩阵
            vert_data = np.full(vel_data.shape, nodata_val if nodata_val else np.nan, dtype=vel_data.dtype)

            # 处理 NoData
            if nodata_val is not None:
                is_data = (vel_data != nodata_val) & (~np.isnan(vel_data))
                valid_mask = valid_mask & is_data
            else:
                # 如果没有nodata值，也要过滤掉NaN
                is_data = ~np.isnan(vel_data)
                valid_mask = valid_mask & is_data

            # 只在有效像素上进行除法运算
            vert_data[valid_mask] = vel_data[valid_mask] / cos_theta[valid_mask]

            # 4. 写入新文件
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(vert_data, 1)

            print(f"  -> 完成: {os.path.basename(out_path)}")

    except Exception as e:
        print(f"  [Error] 处理出错: {e}")


def main():
    # 遍历目录
    for root, dirs, files in os.walk(ROOT_DIR):
        for filename in files:
            # 寻找速度文件
            if filename.startswith(VEL_PREFIX) and filename.endswith(SUFFIX):

                # 1. 解析 ID
                base_part = filename[len(VEL_PREFIX):]
                id_str = base_part[:-len(SUFFIX)]

                vel_full_path = os.path.join(root, filename)

                # 2. 确定入射角文件路径
                # 策略: 优先找 ia_ID.tif，找不到则找 incidenceAngle.tif

                # 路径 A: 匹配 ID 的文件名 (ia_17_113.tif)
                ia_name_pattern = f"{IA_PREFIX}{id_str}{SUFFIX}"
                ia_path_pattern = os.path.join(root, ia_name_pattern)

                # 路径 B: 固定文件名 (incidenceAngle.tif)
                ia_path_fixed = os.path.join(root, FIXED_IA_NAME)

                final_ia_path = None

                if os.path.exists(ia_path_pattern):
                    final_ia_path = ia_path_pattern
                elif os.path.exists(ia_path_fixed):
                    final_ia_path = ia_path_fixed

                # 3. 执行处理
                if final_ia_path:
                    # 构造输出文件名
                    out_filename = filename.replace(SUFFIX, OUT_SUFFIX)
                    out_full_path = os.path.join(root, out_filename)

                    # 如果输出文件已存在，跳过
                    if not os.path.exists(out_full_path):
                        process_tif_pair(vel_full_path, final_ia_path, out_full_path)
                    else:
                        print(f"跳过 (已存在): {out_filename}")
                else:
                    # 如果两个都找不到，打印提示（可选）
                    # print(f"  [Skip] 找不到对应的入射角文件 (ID版或通用版都不存在): {filename}")
                    pass


if __name__ == "__main__":
    main()