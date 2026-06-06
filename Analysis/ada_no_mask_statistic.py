# -*- coding: utf-8 -*-
import os
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import numpy as np
import logging

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
OUT_DIR = r"D:\BaiduSyncdisk\课题首要\Mosaic_InSAR\photos\analysis\ada_mask"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt='%H:%M:%S'
)


# =========================================================
# 1. 核心数学工具库
# =========================================================
def reproject_to_grid(src, profile, resampling=Resampling.bilinear):
    arr = np.empty((profile["height"], profile["width"]), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=arr,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        resampling=resampling,
        dst_nodata=np.nan
    )
    return arr


def high_coh_mask_adaptive(arr, threshold, frac, size):
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


def get_adaptive_mask(a1, a2, level):
    if level == 0:
        # L0: Strict (0.75/0.65) 带空间滤波的自适应
        m1 = high_coh_mask_adaptive(a1, 0.75, 0.65, 7)
        m2 = high_coh_mask_adaptive(a2, 0.75, 0.65, 7)
        return m1 & m2
    elif level == 1:
        # L1: Loose (>=0.40)
        m1 = (a1 >= 0.40)
        m2 = (a2 >= 0.40)
        return m1 & m2
    else:
        # L2: Fallback (>=0.10)
        m1 = (a1 >= 0.10) & np.isfinite(a1)
        m2 = (a2 >= 0.10) & np.isfinite(a2)
        return m1 & m2


# =========================================================
# 2. 主处理逻辑
# =========================================================
if __name__ == "__main__":
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

    num_images = len(inputs)
    stats_report = []

    for i in range(num_images):
        for j in range(i + 1, num_images):
            img1 = inputs[i]
            img2 = inputs[j]

            name1 = os.path.basename(img1['vel']).replace('.tif', '')
            name2 = os.path.basename(img2['vel']).replace('.tif', '')
            pair_tag = f"{name1}_AND_{name2}"

            try:
                with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2:
                    xl = max(src1.bounds.left, src2.bounds.left)
                    xr = min(src1.bounds.right, src2.bounds.right)
                    yb = max(src1.bounds.bottom, src2.bounds.bottom)
                    yt = min(src1.bounds.top, src2.bounds.top)

                    if xr <= xl or yt <= yb:
                        continue

                    width = int((xr - xl) / src1.res[0])
                    height = int((yt - yb) / src1.res[1])
                    if width < 3 or height < 3: continue

                    logging.info(f"Analyzing overlap: {pair_tag}")

                    transform = from_origin(xl, yt, src1.res[0], src1.res[1])
                    prof = {'driver': 'GTiff', 'height': height, 'width': width, 'transform': transform,
                            'crs': src1.crs}

                    with rasterio.open(img1['coh']) as c_src1, rasterio.open(img2['coh']) as c_src2:
                        v1_arr = reproject_to_grid(src1, prof)
                        v2_arr = reproject_to_grid(src2, prof)
                        c1_arr = np.nan_to_num(reproject_to_grid(c_src1, prof))
                        c2_arr = np.nan_to_num(reproject_to_grid(c_src2, prof))
            except Exception as e:
                logging.warning(f"Failed to process pair {pair_tag}: {e}")
                continue

            valid_mask = np.isfinite(v1_arr) & np.isfinite(v2_arr)

            # 初始化记录
            pair_stats = {"Pair": pair_tag}

            # --- 1. 固定阈值统计 (直接像素级判断) ---
            fixed_mask = (c1_arr >= 0.75) & (c2_arr >= 0.75) & valid_mask
            pair_stats["Fixed_0.75"] = int(np.sum(fixed_mask))

            # --- 2. 原始自适应级别统计 (L0-L3) ---
            for lvl in [0, 1, 2, 3]:
                if lvl == 3:
                    tmp_mask = (c1_arr > 0.1) & (c2_arr > 0.1)
                else:
                    tmp_mask = get_adaptive_mask(c1_arr, c2_arr, lvl)

                tmp_mask &= valid_mask
                pair_stats[f"L{lvl}"] = int(np.sum(tmp_mask))

            stats_report.append(pair_stats)

    # =======================================================
    # 汇总输出到 CSV
    # =======================================================
    csv_filename = f"{target_year}.csv"
    stats_file = os.path.join(OUT_DIR, csv_filename)

    with open(stats_file, 'w', encoding='utf-8') as f:
        # 表头：包含固定阈值和自适应各级
        f.write("Pair,Fixed_0.75,L0_Strict(0.75_Ada),L1_Loose(0.40),L2_Fallback(0.10),L3_Desperate(>0.1)\n")

        for st in stats_report:
            f.write(f"{st['Pair']},{st['Fixed_0.75']},{st['L0']},{st['L1']},{st['L2']},{st['L3']}\n")

    logging.info("=" * 65)
    logging.info(f"Statistics successfully saved to: {stats_file}")
    logging.info("ALL TASKS COMPLETED.")