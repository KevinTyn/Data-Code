# -*- coding: utf-8 -*-
import rasterio
import numpy as np
from scipy.ndimage import median_filter
import logging
import os

# =========================
# GPU 加速库检查与回退
# =========================
try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve, percentile_filter

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    from scipy.ndimage import convolve as scipy_convolve
    from scipy.ndimage import percentile_filter as scipy_percentile

# =========================
# 配置与常量
# =========================
NODATA_VALUE = -9999.0
OUT_DIR = r"H:\InSAR_Mosaic_Project"

# 输出清洗后文件的目录
CLEAN_DIR = os.path.join(OUT_DIR, "Cleaned_Velocity")

if not os.path.exists(CLEAN_DIR):
    os.makedirs(CLEAN_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt='%H:%M:%S'
)


# =========================================================
# 工具函数
# =========================================================

def write_with_nodata(arr, filename, out_profile):
    """将处理后的数组写入 GeoTIFF"""
    arr_out = np.where(np.isnan(arr), NODATA_VALUE, arr).astype(np.float32)
    out_profile.update(dtype=rasterio.float32, nodata=NODATA_VALUE)
    with rasterio.open(filename, "w", **out_profile) as dst:
        dst.write(arr_out, 1)


def high_coh_mask_adaptive(arr, threshold, frac, size):
    """自适应高相干性掩膜提取 (GPU/CPU 自动切换)"""
    if HAS_GPU:
        x = cp.array(arr)
        valid = ~cp.isnan(x)
        kernel = cp.ones((size, size), dtype=cp.float32)
        count_total = convolve(valid.astype(cp.float32), kernel, mode="constant", cval=0.0)
        count_high = convolve((x > threshold).astype(cp.float32), kernel, mode="constant", cval=0.0)
        coverage = count_high / cp.maximum(count_total, 1.0)
        x_filled = cp.where(valid, x, threshold - 1e-6)
        med = percentile_filter(x_filled, percentile=50, size=size, mode="nearest")
        mask = (coverage >= frac) & (med >= threshold) & (x >= threshold)
        return cp.asnumpy(mask)
    else:
        x = np.nan_to_num(arr, nan=-9999)
        valid = (arr != -9999) & np.isfinite(arr)
        kernel = np.ones((size, size), dtype=np.float32)
        count_total = scipy_convolve(valid.astype(np.float32), kernel, mode="constant", cval=0.0)
        count_high = scipy_convolve((x > threshold).astype(np.float32), kernel, mode="constant", cval=0.0)
        coverage = count_high / np.maximum(count_total, 1.0)
        med = scipy_percentile(x, percentile=50, size=size, mode="nearest")
        mask = (coverage >= frac) & (med >= threshold) & (x >= threshold)
        return mask


def get_single_adaptive_mask(coh_arr, level):
    """
    分级相干性掩膜策略
    Level 0: 严格 (Strict) - 自适应窗口，中心及周围需满足 0.75 以上
    Level 1: 宽松 (Loose) - 绝对阈值 0.40
    Level 2: 兜底 (Fallback) - 绝对阈值 0.10
    """
    if level == 0:
        mask = high_coh_mask_adaptive(coh_arr, threshold=0.75, frac=0.65, size=3)
        return mask, "Strict (Adaptive 0.75/0.65)"
    elif level == 1:
        mask = (coh_arr >= 0.40) & np.isfinite(coh_arr)
        return mask, "Loose (>=0.40)"
    else:
        mask = (coh_arr >= 0.10) & np.isfinite(coh_arr)
        return mask, "Fallback (>=0.10)"


def process_single_image(vel_path, coh_path, out_vel_path, out_coh_path, out_mask_path, min_points_thresh=5000,
                         outlier_tolerance=0.05):
    """
    处理单幅影像：分级提取相干点 -> 局部中值滤波剔除粗差 -> 同步导出速度、相干性与掩膜
    """
    with rasterio.open(vel_path) as src_v, rasterio.open(coh_path) as src_c:
        vel = src_v.read(1)
        coh = src_c.read(1)
        prof_v = src_v.profile
        prof_c = src_c.profile

    valid_base = np.isfinite(vel) & (vel != NODATA_VALUE)
    total_valid = np.sum(valid_base)

    if total_valid == 0:
        logging.warning(f"  -> No valid data in {os.path.basename(vel_path)}")
        return

    # 1. 自适应降级寻找合适的相干性掩膜
    coh_mask = None
    applied_level_desc = ""

    for lvl in [0, 1, 2]:
        tmp_mask, desc = get_single_adaptive_mask(coh, lvl)
        combined_tmp = valid_base & tmp_mask
        pts_count = np.sum(combined_tmp)

        if pts_count >= min_points_thresh:
            coh_mask = combined_tmp
            applied_level_desc = desc
            logging.info(f"  -> File: {os.path.basename(vel_path)} | Mode: {desc} | Points: {pts_count}")
            break

    # 如果最宽松的标准都没达到最小点数，直接用兜底模式的掩膜
    if coh_mask is None:
        coh_mask, applied_level_desc = get_single_adaptive_mask(coh, 2)
        coh_mask = valid_base & coh_mask
        logging.warning(
            f"  -> File: {os.path.basename(vel_path)} | Forced to Fallback Mode | Points: {np.sum(coh_mask)}")

    # 2. 粗差剔除 (空间离群值剔除)
    # 用 0.0 填充无效区域，避免 nan 干扰中值计算
    vel_filled = np.where(coh_mask, vel, 0.0)
    vel_med = median_filter(vel_filled, size=7)

    # 判定条件：属于相干点，且偏离局部中值过大
    diff = np.abs(vel - vel_med)
    outlier_mask = (diff > outlier_tolerance) & coh_mask

    # 最终掩膜：是高相干点，且不是粗差
    final_mask = coh_mask & (~outlier_mask)
    final_pts = np.sum(final_mask)

    # 3. 生成干净的输出数组 (包含掩膜)
    vel_clean = np.full_like(vel, np.nan, dtype=np.float32)
    coh_clean = np.full_like(coh, np.nan, dtype=np.float32)
    mask_export = np.full_like(vel, np.nan, dtype=np.float32) # 初始化掩膜输出数组

    vel_clean[final_mask] = vel[final_mask]
    coh_clean[final_mask] = coh[final_mask]
    mask_export[final_mask] = 1.0 # 掩膜有效区域设为 1.0，其余为 NaN (最终会被写为 NoData)

    # 4. 写入文件
    write_with_nodata(vel_clean, out_vel_path, prof_v)
    write_with_nodata(coh_clean, out_coh_path, prof_c)
    write_with_nodata(mask_export, out_mask_path, prof_v) # 写入掩膜 TIF

    logging.info(f"     Removed Outliers: {np.sum(outlier_mask)} | Final Retained: {final_pts}")
    logging.info("-" * 50)


if __name__ == "__main__":
    # --- 输入数据 ---
    target_year = 19

    inputs = [
        {'id': 0, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_126\vel_{target_year}_113_126.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_113_126\coh_{target_year}_113_126.tif"},
        {'id': 1, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_121\vel_{target_year}_113_121.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_113_121\coh_{target_year}_113_121.tif"},
        {'id': 2, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_116\vel_{target_year}_113_116.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_113_116\coh_{target_year}_113_116.tif"},
        {'id': 3, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_111\vel_{target_year}_113_111.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_113_111\coh_{target_year}_113_111.tif"},
        {'id': 4, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_121\vel_{target_year}_11_121.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_11_121\coh_{target_year}_11_121.tif"},
        {'id': 5, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_116\vel_{target_year}_11_116.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_11_116\coh_{target_year}_11_116.tif"},
        {'id': 6, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_111\vel_{target_year}_11_111.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_11_111\coh_{target_year}_11_111.tif"},
        {'id': 7, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_127\vel_{target_year}_40_127.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_40_127\coh_{target_year}_40_127.tif"},
        {'id': 8, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_122\vel_{target_year}_40_122.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_40_122\coh_{target_year}_40_122.tif"},
        {'id': 9, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_117\vel_{target_year}_40_117.tif",
         'coh': fr"E:\shanxi\{target_year}\{target_year}_40_117\coh_{target_year}_40_117.tif"},
    ]

    logging.info("====== Starting Adaptive High-Coherence Extraction & Outlier Removal ======")

    for item in inputs:
        vel_in = item['vel']
        coh_in = item['coh']

        base_vel_name = os.path.basename(vel_in)
        base_coh_name = os.path.basename(coh_in)

        # 增加掩膜的文件输出名
        out_vel_path = os.path.join(CLEAN_DIR, f"Cleaned_{base_vel_name}")
        out_coh_path = os.path.join(CLEAN_DIR, f"Cleaned_{base_coh_name}")
        out_mask_path = os.path.join(CLEAN_DIR, f"Mask_{base_vel_name}")

        if os.path.exists(vel_in) and os.path.exists(coh_in):
            process_single_image(
                vel_path=vel_in,
                coh_path=coh_in,
                out_vel_path=out_vel_path,
                out_coh_path=out_coh_path,
                out_mask_path=out_mask_path,  # 传入新增的参数
                min_points_thresh=5000,
                outlier_tolerance=0.05
            )
        else:
            logging.error(f"  -> File not found: {vel_in} or {coh_in}")

    logging.info("====== All Tasks Completed. Check 'Cleaned_Velocity' folder. ======")