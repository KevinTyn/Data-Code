import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import generic_filter
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline
import os

# 导入 GPU 加速库
import cupy as cp
from cupyx.scipy.ndimage import uniform_filter

# ==========================================
# 0. 配置部分
# ==========================================
# path_groups = [
#     (r"E:\shanxi\18\18_11_111\vel_18_11_111.tif", r"E:\shanxi\18\18_11_116\vel_18_11_116.tif",
#      r"E:\shanxi\18\18_11_111\coh_18_11_111.tif", r"E:\shanxi\18\18_11_116\coh_18_11_116.tif"),
#     (r"E:\shanxi\18\18_11_116\vel_18_11_116.tif", r"E:\shanxi\18\18_11_121\vel_18_11_121.tif",
#      r"E:\shanxi\18\18_11_116\coh_18_11_116.tif", r"E:\shanxi\18\18_11_121\coh_18_11_121.tif"),
#     (r"E:\shanxi\18\18_40_117\vel_18_40_117.tif", r"E:\shanxi\18\18_40_122\vel_18_40_122.tif",
#      r"E:\shanxi\18\18_40_117\coh_18_40_117.tif", r"E:\shanxi\18\18_40_122\coh_18_40_122.tif"),
#     (r"E:\shanxi\18\18_40_122\vel_18_40_122.tif", r"E:\shanxi\18\18_40_127\vel_18_40_127.tif",
#      r"E:\shanxi\18\18_40_122\coh_18_40_122.tif", r"E:\shanxi\18\18_40_127\coh_18_40_127.tif"),
#     (r"E:\shanxi\18\18_113_111\vel_18_113_111.tif", r"E:\shanxi\18\18_113_116\vel_18_113_116.tif",
#      r"E:\shanxi\18\18_113_111\coh_18_113_111.tif", r"E:\shanxi\18\18_113_116\coh_18_113_116.tif"),
#     (r"E:\shanxi\18\18_113_116\vel_18_113_116.tif", r"E:\shanxi\18\18_113_121\vel_18_113_121.tif",
#      r"E:\shanxi\18\18_113_116\coh_18_113_116.tif", r"E:\shanxi\18\18_113_121\coh_18_113_121.tif"),
#     (r"E:\shanxi\18\18_113_121\vel_18_113_121.tif", r"E:\shanxi\18\18_113_126\vel_18_113_126.tif",
#      r"E:\shanxi\18\18_113_121\coh_18_113_121.tif", r"E:\shanxi\18\18_113_126\coh_18_113_126.tif"),
#     (r"E:\shanxi\18\18_11_111\vel_18_11_111.tif", r"E:\shanxi\18\18_113_111\vel_18_113_111.tif",
#      r"E:\shanxi\18\18_11_111\coh_18_11_111.tif", r"E:\shanxi\18\18_113_111\coh_18_113_111.tif"),
#     (r"E:\shanxi\18\18_11_116\vel_18_11_116.tif", r"E:\shanxi\18\18_113_116\vel_18_113_116.tif",
#      r"E:\shanxi\18\18_11_116\coh_18_11_116.tif", r"E:\shanxi\18\18_113_116\coh_18_113_116.tif"),
#     (r"E:\shanxi\18\18_11_121\vel_18_11_121.tif", r"E:\shanxi\18\18_113_121\vel_18_113_121.tif",
#      r"E:\shanxi\18\18_11_121\coh_18_11_121.tif", r"E:\shanxi\18\18_113_121\coh_18_113_121.tif"),
#     (r"E:\shanxi\18\18_113_116\vel_18_113_116.tif", r"E:\shanxi\18\18_40_117\vel_18_40_117.tif",
#      r"E:\shanxi\18\18_113_116\coh_18_113_116.tif", r"E:\shanxi\18\18_40_117\coh_18_40_117.tif"),
#     (r"E:\shanxi\18\18_113_121\vel_18_113_121.tif", r"E:\shanxi\18\18_40_122\vel_18_40_122.tif",
#      r"E:\shanxi\18\18_113_121\coh_18_113_121.tif", r"E:\shanxi\18\18_40_122\coh_18_40_122.tif"),
#     (r"E:\shanxi\18\18_113_126\vel_18_113_126.tif", r"E:\shanxi\18\18_40_127\vel_18_40_127.tif",
#      r"E:\shanxi\18\18_113_126\coh_18_113_126.tif", r"E:\shanxi\18\18_40_127\coh_18_40_127.tif")
# ]
# path_groups = [
#     (r"E:\shanxi\17\17_11_111\vel_17_11_111.tif", r"E:\shanxi\17\17_11_116\vel_17_11_116.tif",
#      r"E:\shanxi\17\17_11_111\coh_17_11_111.tif", r"E:\shanxi\17\17_11_116\coh_17_11_116.tif"),
#     (r"E:\shanxi\17\17_11_116\vel_17_11_116.tif", r"E:\shanxi\17\17_11_121\vel_17_11_121.tif",
#      r"E:\shanxi\17\17_11_116\coh_17_11_116.tif", r"E:\shanxi\17\17_11_121\coh_17_11_121.tif"),
#     (r"E:\shanxi\17\17_40_117\vel_17_40_117.tif", r"E:\shanxi\17\17_40_122\vel_17_40_122.tif",
#      r"E:\shanxi\17\17_40_117\coh_17_40_117.tif", r"E:\shanxi\17\17_40_122\coh_17_40_122.tif"),
#     (r"E:\shanxi\17\17_40_122\vel_17_40_122.tif", r"E:\shanxi\17\17_40_127\vel_17_40_127.tif",
#      r"E:\shanxi\17\17_40_122\coh_17_40_122.tif", r"E:\shanxi\17\17_40_127\coh_17_40_127.tif"),
#     (r"E:\shanxi\17\17_113_111\vel_17_113_111.tif", r"E:\shanxi\17\17_113_116\vel_17_113_116.tif",
#      r"E:\shanxi\17\17_113_111\coh_17_113_111.tif", r"E:\shanxi\17\17_113_116\coh_17_113_116.tif"),
#     (r"E:\shanxi\17\17_113_116\vel_17_113_116.tif", r"E:\shanxi\17\17_113_121\vel_17_113_121.tif",
#      r"E:\shanxi\17\17_113_116\coh_17_113_116.tif", r"E:\shanxi\17\17_113_121\coh_17_113_121.tif"),
#     (r"E:\shanxi\17\17_113_121\vel_17_113_121.tif", r"E:\shanxi\17\17_113_126\vel_17_113_126.tif",
#      r"E:\shanxi\17\17_113_121\coh_17_113_121.tif", r"E:\shanxi\17\17_113_126\coh_17_113_126.tif"),
#     (r"E:\shanxi\17\17_11_111\vel_17_11_111.tif", r"E:\shanxi\17\17_113_111\vel_17_113_111.tif",
#      r"E:\shanxi\17\17_11_111\coh_17_11_111.tif", r"E:\shanxi\17\17_113_111\coh_17_113_111.tif"),
#     (r"E:\shanxi\17\17_11_116\vel_17_11_116.tif", r"E:\shanxi\17\17_113_116\vel_17_113_116.tif",
#      r"E:\shanxi\17\17_11_116\coh_17_11_116.tif", r"E:\shanxi\17\17_113_116\coh_17_113_116.tif"),
#     (r"E:\shanxi\17\17_11_121\vel_17_11_121.tif", r"E:\shanxi\17\17_113_121\vel_17_113_121.tif",
#      r"E:\shanxi\17\17_11_121\coh_17_11_121.tif", r"E:\shanxi\17\17_113_121\coh_17_113_121.tif"),
#     (r"E:\shanxi\17\17_113_116\vel_17_113_116.tif", r"E:\shanxi\17\17_40_117\vel_17_40_117.tif",
#      r"E:\shanxi\17\17_113_116\coh_17_113_116.tif", r"E:\shanxi\17\17_40_117\coh_17_40_117.tif"),
#     (r"E:\shanxi\17\17_113_121\vel_17_113_121.tif", r"E:\shanxi\17\17_40_122\vel_17_40_122.tif",
#      r"E:\shanxi\17\17_113_121\coh_17_113_121.tif", r"E:\shanxi\17\17_40_122\coh_17_40_122.tif"),
#     (r"E:\shanxi\17\17_113_126\vel_17_113_126.tif", r"E:\shanxi\17\17_40_127\vel_17_40_127.tif",
#      r"E:\shanxi\17\17_113_126\coh_17_113_126.tif", r"E:\shanxi\17\17_40_127\coh_17_40_127.tif")
# ]
path_groups = [
    (r"E:\shanxi\19\19_11_111\vel_19_11_111.tif", r"E:\shanxi\19\19_11_116\vel_19_11_116.tif",
     r"E:\shanxi\19\19_11_111\coh_19_11_111.tif", r"E:\shanxi\19\19_11_116\coh_19_11_116.tif"),
    (r"E:\shanxi\19\19_11_116\vel_19_11_116.tif", r"E:\shanxi\19\19_11_121\vel_19_11_121.tif",
     r"E:\shanxi\19\19_11_116\coh_19_11_116.tif", r"E:\shanxi\19\19_11_121\coh_19_11_121.tif"),
    (r"E:\shanxi\19\19_40_117\vel_19_40_117.tif", r"E:\shanxi\19\19_40_122\vel_19_40_122.tif",
     r"E:\shanxi\19\19_40_117\coh_19_40_117.tif", r"E:\shanxi\19\19_40_122\coh_19_40_122.tif"),
    (r"E:\shanxi\19\19_40_122\vel_19_40_122.tif", r"E:\shanxi\19\19_40_127\vel_19_40_127.tif",
     r"E:\shanxi\19\19_40_122\coh_19_40_122.tif", r"E:\shanxi\19\19_40_127\coh_19_40_127.tif"),
    (r"E:\shanxi\19\19_113_111\vel_19_113_111.tif", r"E:\shanxi\19\19_113_116\vel_19_113_116.tif",
     r"E:\shanxi\19\19_113_111\coh_19_113_111.tif", r"E:\shanxi\19\19_113_116\coh_19_113_116.tif"),
    (r"E:\shanxi\19\19_113_116\vel_19_113_116.tif", r"E:\shanxi\19\19_113_121\vel_19_113_121.tif",
     r"E:\shanxi\19\19_113_116\coh_19_113_116.tif", r"E:\shanxi\19\19_113_121\coh_19_113_121.tif"),
    (r"E:\shanxi\19\19_113_121\vel_19_113_121.tif", r"E:\shanxi\19\19_113_126\vel_19_113_126.tif",
     r"E:\shanxi\19\19_113_121\coh_19_113_121.tif", r"E:\shanxi\19\19_113_126\coh_19_113_126.tif"),
    (r"E:\shanxi\19\19_11_111\vel_19_11_111.tif", r"E:\shanxi\19\19_113_111\vel_19_113_111.tif",
     r"E:\shanxi\19\19_11_111\coh_19_11_111.tif", r"E:\shanxi\19\19_113_111\coh_19_113_111.tif"),
    (r"E:\shanxi\19\19_11_116\vel_19_11_116.tif", r"E:\shanxi\19\19_113_116\vel_19_113_116.tif",
     r"E:\shanxi\19\19_11_116\coh_19_11_116.tif", r"E:\shanxi\19\19_113_116\coh_19_113_116.tif"),
    (r"E:\shanxi\19\19_11_121\vel_19_11_121.tif", r"E:\shanxi\19\19_113_121\vel_19_113_121.tif",
     r"E:\shanxi\19\19_11_121\coh_19_11_121.tif", r"E:\shanxi\19\19_113_121\coh_19_113_121.tif"),
    (r"E:\shanxi\19\19_113_116\vel_19_113_116.tif", r"E:\shanxi\19\19_40_117\vel_19_40_117.tif",
     r"E:\shanxi\19\19_113_116\coh_19_113_116.tif", r"E:\shanxi\19\19_40_117\coh_19_40_117.tif"),
    (r"E:\shanxi\19\19_113_121\vel_19_113_121.tif", r"E:\shanxi\19\19_40_122\vel_19_40_122.tif",
     r"E:\shanxi\19\19_113_121\coh_19_113_121.tif", r"E:\shanxi\19\19_40_122\coh_19_40_122.tif"),
    (r"E:\shanxi\19\19_113_126\vel_19_113_126.tif", r"E:\shanxi\19\19_40_127\vel_19_40_127.tif",
     r"E:\shanxi\19\19_113_126\coh_19_113_126.tif", r"E:\shanxi\19\19_40_127\coh_19_40_127.tif")
]
# path_groups = [
#     (r"E:\shanxi\2123\2123_11_111\vel_2123_11_111.tif", r"E:\shanxi\2123\2123_11_116\vel_2123_11_116.tif",
#      r"E:\shanxi\2123\2123_11_111\coh_2123_11_111.tif", r"E:\shanxi\2123\2123_11_116\coh_2123_11_116.tif"),
#     (r"E:\shanxi\2123\2123_11_116\vel_2123_11_116.tif", r"E:\shanxi\2123\2123_11_121\vel_2123_11_121.tif",
#      r"E:\shanxi\2123\2123_11_116\coh_2123_11_116.tif", r"E:\shanxi\2123\2123_11_121\coh_2123_11_121.tif"),
#     (r"E:\shanxi\2123\2123_40_117\vel_2123_40_117.tif", r"E:\shanxi\2123\2123_40_122\vel_2123_40_122.tif",
#      r"E:\shanxi\2123\2123_40_117\coh_2123_40_117.tif", r"E:\shanxi\2123\2123_40_122\coh_2123_40_122.tif"),
#     (r"E:\shanxi\2123\2123_40_122\vel_2123_40_122.tif", r"E:\shanxi\2123\2123_40_127\vel_2123_40_127.tif",
#      r"E:\shanxi\2123\2123_40_122\coh_2123_40_122.tif", r"E:\shanxi\2123\2123_40_127\coh_2123_40_127.tif"),
#     (r"E:\shanxi\2123\2123_113_111\vel_2123_113_111.tif", r"E:\shanxi\2123\2123_113_116\vel_2123_113_116.tif",
#      r"E:\shanxi\2123\2123_113_111\coh_2123_113_111.tif", r"E:\shanxi\2123\2123_113_116\coh_2123_113_116.tif"),
#     (r"E:\shanxi\2123\2123_113_116\vel_2123_113_116.tif", r"E:\shanxi\2123\2123_113_121\vel_2123_113_121.tif",
#      r"E:\shanxi\2123\2123_113_116\coh_2123_113_116.tif", r"E:\shanxi\2123\2123_113_121\coh_2123_113_121.tif"),
#     (r"E:\shanxi\2123\2123_113_121\vel_2123_113_121.tif", r"E:\shanxi\2123\2123_113_126\vel_2123_113_126.tif",
#      r"E:\shanxi\2123\2123_113_121\coh_2123_113_121.tif", r"E:\shanxi\2123\2123_113_126\coh_2123_113_126.tif"),
#     (r"E:\shanxi\2123\2123_11_111\vel_2123_11_111.tif", r"E:\shanxi\2123\2123_113_111\vel_2123_113_111.tif",
#      r"E:\shanxi\2123\2123_11_111\coh_2123_11_111.tif", r"E:\shanxi\2123\2123_113_111\coh_2123_113_111.tif"),
#     (r"E:\shanxi\2123\2123_11_116\vel_2123_11_116.tif", r"E:\shanxi\2123\2123_113_116\vel_2123_113_116.tif",
#      r"E:\shanxi\2123\2123_11_116\coh_2123_11_116.tif", r"E:\shanxi\2123\2123_113_116\coh_2123_113_116.tif"),
#     (r"E:\shanxi\2123\2123_11_121\vel_2123_11_121.tif", r"E:\shanxi\2123\2123_113_121\vel_2123_113_121.tif",
#      r"E:\shanxi\2123\2123_11_121\coh_2123_11_121.tif", r"E:\shanxi\2123\2123_113_121\coh_2123_113_121.tif"),
#     (r"E:\shanxi\2123\2123_113_116\vel_2123_113_116.tif", r"E:\shanxi\2123\2123_40_117\vel_2123_40_117.tif",
#      r"E:\shanxi\2123\2123_113_116\coh_2123_113_116.tif", r"E:\shanxi\2123\2123_40_117\coh_2123_40_117.tif"),
#     (r"E:\shanxi\2123\2123_113_121\vel_2123_113_121.tif", r"E:\shanxi\2123\2123_40_122\vel_2123_40_122.tif",
#      r"E:\shanxi\2123\2123_113_121\coh_2123_113_121.tif", r"E:\shanxi\2123\2123_40_122\coh_2123_40_122.tif"),
#     (r"E:\shanxi\2123\2123_113_126\vel_2123_113_126.tif", r"E:\shanxi\2123\2123_40_127\vel_2123_40_127.tif",
#      r"E:\shanxi\2123\2123_113_126\coh_2123_113_126.tif", r"E:\shanxi\2123\2123_40_127\coh_2123_40_127.tif")
# ]

# --- 输出保存路径 ---
save_dir = r"E:\shanxi\19\4\output_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"创建输出目录: {save_dir}")

# --- 绘图全局设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['svg.fonttype'] = 'none'

plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2


# ==========================================
# 1. 工具函数
# ==========================================
def reproject_to_grid(src, profile):
    arr = np.empty((profile["height"], profile["width"]), dtype=src.dtypes[0])
    reproject(
        source=rasterio.band(src, 1),
        destination=arr,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        resampling=Resampling.nearest
    )
    return np.ma.masked_equal(arr, src.nodata)


def safe_nanstd(values):
    vals = values[~np.isnan(values)]
    if len(vals) > 1:
        return np.std(vals)
    else:
        return np.nan


def get_clean_vmax_and_ticks(max_val):
    if max_val <= 0:
        return 10.0, np.linspace(0, 10, 5), 0, 1

    exponent = int(np.floor(np.log10(max_val)))
    scale = 10 ** exponent
    scaled_max = max_val / scale

    candidates = [1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.8, 6.0, 8.0, 10.0]
    chosen_scaled_vmax = 10.0
    for c in candidates:
        if c >= scaled_max:
            chosen_scaled_vmax = c
            break

    vmax = chosen_scaled_vmax * scale
    ticks = np.linspace(0, vmax, 5)
    return vmax, ticks, exponent, scale


# ==========================================
# 核心逻辑：两阶段处理
# 阶段 1：遍历处理所有组，记录全局最大值并缓存
# ==========================================
print("\n========== [阶段 1/2] 开始数据处理与全局最大值提取 ==========")

total_groups = len(path_groups)
plot_data_cache = []
global_max_hexbin = 0.0
global_max_line = 0.0

for idx, paths in enumerate(path_groups, start=1):
    tif1, tif2, coh1, coh2 = paths
    print(f"\n[{idx}/{total_groups}] 正在处理组合:\n 1: {tif1}\n 2: {tif2}")

    if not all(os.path.exists(p) for p in paths):
        print(f"警告：该组中包含不存在的文件，跳过此组...")
        continue

    print("  -> 正在读取和对齐栅格数据...")
    src1, src2 = rasterio.open(tif1), rasterio.open(tif2)
    src_coh1, src_coh2 = rasterio.open(coh1), rasterio.open(coh2)

    xmin = max(src1.bounds.left, src2.bounds.left)
    ymin = max(src1.bounds.bottom, src2.bounds.bottom)
    xmax = min(src1.bounds.right, src2.bounds.right)
    ymax = min(src1.bounds.top, src2.bounds.top)

    res, crs = src1.res, src1.crs
    width = int((xmax - xmin) / res[0])
    height = int((ymax - ymin) / res[1])
    transform = from_origin(xmin, ymax, *res)

    profile = src1.profile.copy()
    profile.update({"height": height, "width": width, "transform": transform, "crs": crs})

    arr1 = reproject_to_grid(src1, profile)
    arr2 = reproject_to_grid(src2, profile)
    coh_arr1 = reproject_to_grid(src_coh1, profile)
    coh_arr2 = reproject_to_grid(src_coh2, profile)

    mask = (~arr1.mask) & (~arr2.mask)
    arr1 = np.where(mask, arr1, profile["nodata"])
    arr2 = np.where(mask, arr2, profile["nodata"])
    coh_arr1 = np.where(mask, coh_arr1, profile["nodata"])
    coh_arr2 = np.where(mask, coh_arr2, profile["nodata"])

    print("  -> 正在执行 3-Sigma 异常值剔除(GPU加速)...")
    arr1_cp = cp.asarray(arr1)
    arr2_cp = cp.asarray(arr2)

    diff_cp = cp.where((arr1_cp != profile["nodata"]) & (arr2_cp != profile["nodata"]), arr1_cp - arr2_cp, cp.nan)
    mu_cp = cp.nanmean(diff_cp)
    sigma_cp = cp.nanstd(diff_cp)
    mask_outlier_cp = cp.abs(diff_cp - mu_cp) > 3 * sigma_cp

    arr1 = cp.asnumpy(cp.where(mask_outlier_cp, profile["nodata"], arr1_cp))
    arr2 = cp.asnumpy(cp.where(mask_outlier_cp, profile["nodata"], arr2_cp))

    # -------------------------
    # 直接在第一阶段生成并保存图 1
    # -------------------------
    print("  -> 正在生成并保存图表 1 (3D散点与2D场)...")
    fig1 = plt.figure(figsize=(14, 6))
    fig1.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9, wspace=0.25)

    ax1 = fig1.add_subplot(121, projection="3d")


    def add_scatter(arr, profile, label, color, sample=10000):
        rows, cols = np.where(arr != profile["nodata"])
        if len(rows) > sample:
            idx_sample = np.random.choice(len(rows), sample, replace=False)
            rows, cols = rows[idx_sample], cols[idx_sample]
        xs, ys = rasterio.transform.xy(profile["transform"], rows, cols)
        zs = arr[rows, cols]
        ax1.scatter(xs, ys, zs, c=color, s=2, label=label, alpha=0.6)


    add_scatter(arr1, profile, "Velocity 1", "blue")
    add_scatter(arr2, profile, "Velocity 2", "red")
    ax1.set_xlabel("X (m)", labelpad=10)
    ax1.set_ylabel("Y (m)", labelpad=10)
    ax1.set_zlabel("Velocity", labelpad=10)
    ax1.set_title("InSAR Overlap Data (3D Scatter)")
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.legend()

    ax2 = fig1.add_subplot(122)
    ext_left, ext_top = profile['transform'][2], profile['transform'][5]
    ext_right = ext_left + profile['width'] * profile['transform'][0]
    ext_bottom = ext_top + profile['height'] * profile['transform'][4]
    im = ax2.imshow(np.where(arr1 != profile["nodata"], arr1, np.nan),
                    cmap="jet", extent=[ext_left, ext_right, ext_bottom, ext_top], origin='upper')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Velocity")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Velocity Field (2D Colormap)")
    ax2.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    plt.savefig(os.path.join(save_dir, f"1_Velocity_Field_3D_2D_{idx}.svg"), dpi=600)
    plt.close(fig1)

    # -------------------------
    # 提取全局最大值并缓存数据
    # -------------------------
    print("  -> 正在计算局部速率波动(GPU加速)...")
    vel_data = np.where(arr1 != profile["nodata"], arr1, np.nan)
    coh_data = np.where(coh_arr1 != profile["nodata"], coh_arr1, np.nan)

    # GPU 矩阵操作替代 generic_filter
    vel_data_cp = cp.asarray(vel_data)
    valid_mask_cp = cp.isfinite(vel_data_cp)
    vel_filled_cp = cp.where(valid_mask_cp, vel_data_cp, 0)

    # 5x5 窗口处理
    n_valid_cp = uniform_filter(valid_mask_cp.astype(cp.float32), size=5, mode='constant', cval=0) * 25
    sum_x_cp = uniform_filter(vel_filled_cp, size=5, mode='constant', cval=0) * 25
    sum_x2_cp = uniform_filter(vel_filled_cp ** 2, size=5, mode='constant', cval=0) * 25

    mean_x_cp = sum_x_cp / cp.maximum(n_valid_cp, 1)
    mean_x2_cp = sum_x2_cp / cp.maximum(n_valid_cp, 1)
    var_x_cp = cp.maximum(mean_x2_cp - mean_x_cp ** 2, 0)
    std_x_cp = cp.sqrt(var_x_cp)

    std_x_cp = cp.where(n_valid_cp > 1, std_x_cp, cp.nan)
    vel_fluct = cp.asnumpy(std_x_cp)

    valid = (~np.isnan(vel_fluct)) & (~np.isnan(coh_data))
    vel_flat = vel_fluct[valid].ravel()
    coh_flat = coh_data[valid].ravel()

    valid_mask = np.isfinite(coh_flat) & np.isfinite(vel_flat)
    x_data = coh_flat[valid_mask]
    y_data = vel_flat[valid_mask]

    if len(y_data) > 0:
        global_max_hexbin = max(global_max_hexbin, y_data.max())

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means, bin_stds = [], []
    for i in range(len(bins) - 1):
        mask_bin = (coh_flat >= bins[i]) & (coh_flat < bins[i + 1])
        if mask_bin.sum() > 0:
            bin_means.append(vel_flat[mask_bin].mean())
            bin_stds.append(vel_flat[mask_bin].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    valid_means = np.array(bin_means)[~np.isnan(bin_means)]
    valid_stds = np.array(bin_stds)[~np.isnan(bin_stds)]
    if len(valid_means) > 0:
        global_max_line = max(global_max_line, np.max(valid_means + valid_stds))

    plot_data_cache.append({
        'idx': idx,
        'x_data': x_data,
        'y_data': y_data,
        'bin_centers': bin_centers,
        'bin_means': bin_means,
        'bin_stds': bin_stds
    })

global_max_hexbin *= 1.05
global_max_line *= 1.1

print(f"\n全局计算完毕！")
print(f" -> Hexbin 图统一最大 Y 值将被锁定为: {global_max_hexbin:.2f}")
print(f" -> 统计折线图统一最大 Y 值将被锁定为: {global_max_line:.2f}")

# ==========================================
# 阶段 2：使用统一纵坐标与精细刻度，批量生成单图
# ==========================================
print("\n========== [阶段 2/2] 开始批量绘制单组统计图表 ==========")

for data in plot_data_cache:
    idx = data['idx']
    x_data = data['x_data']
    y_data = data['y_data']
    bin_centers = data['bin_centers']
    bin_means = data['bin_means']
    bin_stds = data['bin_stds']

    print(f"正在生成第 [{idx}/{total_groups}] 组的单图...")

    if len(x_data) < 2:
        print(f"  -> [{idx}/{total_groups}] 数据点不足，跳过...")
        continue

    # -------------------------
    # 绘图 2: 高级六边形分布图 (精细修复 Colorbar 与图例)
    # -------------------------
    r, p = pearsonr(x_data, y_data)
    slope, intercept = np.polyfit(x_data, y_data, 1)

    fig2, ax = plt.subplots(figsize=(8, 7))
    fig2.subplots_adjust(left=0.18, right=0.92, top=0.88, bottom=0.15)

    hb = ax.hexbin(x_data, y_data, gridsize=40, cmap='Blues', mincnt=1)
    cb = fig2.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Count', fontsize=20, fontweight='bold', color='black')

    counts = hb.get_array()
    max_count = counts.max() if len(counts) > 0 else 0
    vmax_cb, ticks_cb, exponent, scale = get_clean_vmax_and_ticks(max_count)

    hb.set_clim(0, vmax_cb)
    cb.set_ticks(ticks_cb)


    def custom_cb_formatter(x, pos):
        return f"{x / scale:.1f}"


    cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_cb_formatter))
    cb.ax.tick_params(labelsize=18)

    if exponent != 0:
        cb.ax.set_title(fr'$\times 10^{{{exponent}}}$', fontsize=18, pad=15)

    x_fit = np.linspace(0, 1.0, 100)
    y_fit = slope * x_fit + intercept
    label_text = f'Linear Fit: $y={slope:.2f}x+{intercept:.2f}$\n(Pearson $R={r:.2f}$)'
    ax.plot(x_fit, y_fit, color='#FF3333', linestyle='--', linewidth=2.5, label=label_text)

    ax.set_xlim(left=0, right=1.0)
    ax.set_ylim(bottom=0, top=global_max_hexbin)

    ax.set_xlabel("Coherence (-)", fontsize=24, fontweight='bold', color='black')
    ax.set_ylabel("Local Velocity Fluctuation", fontsize=24, fontweight='bold', color='black')
    ax.tick_params(axis='both', colors='black', labelsize=22)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    ax.grid(True, linestyle="--", alpha=0.5, color='gray')
    ax.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='#F5F5F5', fontsize=20)

    plt.savefig(os.path.join(save_dir, f"2_Hexbin_Plot_{idx}.svg"), dpi=600)
    plt.close(fig2)

    # -------------------------
    # 绘图 3: 分箱统计
    # -------------------------
    fig3 = plt.figure(figsize=(6, 5))
    fig3.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.18)

    ax3 = plt.gca()
    ax3.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='-o', capsize=4, color='black', label='Mean ± Std')
    ax3.set_xlabel("Coherence (binned)", fontweight='bold', fontsize=24)
    ax3.set_ylabel("Local Velocity Fluctuation", fontweight='bold', fontsize=24)
    ax3.set_ylim(bottom=0, top=global_max_line)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax3.set_title("Binned Analysis: Fluctuation Trend", fontweight='bold', fontsize=16)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=14)

    plt.savefig(os.path.join(save_dir, f"3_Binned_Statistics_{idx}.svg"), dpi=600)
    plt.close(fig3)

    # -------------------------
    # 绘图 4: 梯度分析
    # -------------------------
    valid_bins_mask = ~np.isnan(bin_means)
    x_spline = bin_centers[valid_bins_mask]
    y_spline = np.array(bin_means)[valid_bins_mask]

    fig4 = plt.figure(figsize=(6, 5))
    fig4.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.18)

    ax4 = plt.gca()
    if len(x_spline) > 3:
        spline = UnivariateSpline(x_spline, y_spline, s=0.001)
        xx = np.linspace(0, 1, 200)
        yy = spline(xx)
        dy = spline.derivative()(xx)
        idx_max_grad = np.argmax(np.abs(dy))
        x_turn = xx[idx_max_grad]
        y_turn = yy[idx_max_grad]

        ax4.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', capsize=4, label="Binned Data", color='gray',
                     alpha=0.6)
        ax4.plot(xx, yy, 'r-', linewidth=2, label="Spline Fit")
        ax4.axvline(x_turn, color='g', linestyle='--', label=f"Max Gradient at {x_turn:.2f}")
        ax4.scatter([x_turn], [y_turn], color='g', s=80, zorder=5, marker='*')

        ax4.set_xlabel("Coherence", fontweight='bold', fontsize=24)
        ax4.set_ylabel("Local Velocity Fluctuation", fontweight='bold', fontsize=24)
        ax4.set_ylim(bottom=0, top=global_max_line)
        ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax4.set_title("Gradient Analysis of Stability", fontweight='bold', fontsize=16)
        ax4.legend(fontsize=12)
        ax4.grid(True, linestyle="--", alpha=0.5)

        plt.savefig(os.path.join(save_dir, f"4_Gradient_Analysis_{idx}.svg"), dpi=600)
    plt.close(fig4)

# ==========================================
# 阶段 3：全新添加！生成全组合大拼图 (去除重复横纵坐标与主标题)
# ==========================================
if len(plot_data_cache) > 0:
    print("\n========== [阶段 3] 开始合并生成一整个大图 (图2 全组合大拼图) ==========")

    n_rows, n_cols = 4, 4
    fig_all, axs = plt.subplots(n_rows, n_cols, figsize=(25, 23))
    axs = axs.ravel()

    for i, data in enumerate(plot_data_cache):
        ax = axs[i]
        idx = data['idx']
        x_data = data['x_data']
        y_data = data['y_data']

        if len(x_data) < 2:
            ax.set_facecolor('#f0f0f0')
            ax.set_title(f"Group {idx} (no data)", fontsize=16, fontweight='bold', pad=8)
            continue

        r, p = pearsonr(x_data, y_data)
        slope, intercept = np.polyfit(x_data, y_data, 1)

        hb = ax.hexbin(x_data, y_data, gridsize=40, cmap='Blues', mincnt=1)

        counts = hb.get_array()
        max_count = counts.max() if len(counts) > 0 else 0
        vmax_cb, ticks_cb, exponent, scale = get_clean_vmax_and_ticks(max_count)
        hb.set_clim(0, vmax_cb)

        x_fit = np.linspace(0, 1.0, 100)
        y_fit = slope * x_fit + intercept
        label_text = f'$y={slope:.2f}x+{intercept:.2f}$\n$R={r:.2f}$'
        ax.plot(x_fit, y_fit, color='#FF3333', linestyle='--', linewidth=2.0, label=label_text)

        ax.set_xlim(left=0, right=1.0)
        ax.set_ylim(bottom=0, top=global_max_hexbin)
        ax.tick_params(axis='both', colors='black', labelsize=14)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.grid(True, linestyle="--", alpha=0.5, color='gray')

        ax.set_title(f"Group {idx}", fontsize=16, fontweight='bold', pad=8)
        ax.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='#F5F5F5', fontsize=12)

        cb = fig_all.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_ticks(ticks_cb)


        def grid_cb_formatter(x, pos):
            return f"{x / scale:.1f}"


        cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(grid_cb_formatter))
        cb.ax.tick_params(labelsize=12)
        if exponent != 0:
            cb.ax.set_title(fr'$\times 10^{{{exponent}}}$', fontsize=12, pad=10)

    for j in range(len(plot_data_cache), n_rows * n_cols):
        axs[j].axis('off')

    fig_all.supxlabel("Coherence (-)", fontsize=28, fontweight='bold', y=0.02)
    fig_all.supylabel("Local Velocity Fluctuation", fontsize=28, fontweight='bold', x=0.02)

    fig_all.subplots_adjust(left=0.07, right=0.95, top=0.96, bottom=0.07, wspace=0.38, hspace=0.38)

    combined_save_path = os.path.join(save_dir, "2_Combined_Hexbin_All_Groups.svg")
    plt.savefig(combined_save_path, dpi=600)
    plt.close(fig_all)

    print(f"\n✅ 全组合大拼图已成功渲染，保存至: {combined_save_path}")

print("\n=======================================")
print("🎉 优化彻底完成！所有子图的 Count 刻度已转化为绝对等间距，大拼图已在末尾完美输出。")