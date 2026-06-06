# -*- coding: utf-8 -*-
"""
Script 2 (Final V36 - Multi-Dimensional Metrics): InSAR Mosaic Validator
功能：交互画线后，基于线数据进行多维度的精度计算。
新增特性：
1. [Metrics] 增加相关系数 (Pearson r)。
2. [Metrics] 增加平均偏差 (Mean Bias / ME) 评估系统性平移。
3. [Metrics] 增加 <2 mm/yr 占比，评估工程可用容忍度。
4. [Visual] 将相关系数 r 移至图g、h的图外右上角显示，不遮挡数据。
5. [CSV] 将所有新维度同步更新到 Stats_Final.csv 中。
6. [CSV] 增强版 ProfileData 导出，包含 Raw 和 Cor 的所有数据，支持后期完全脱机重绘统计图。
7. [Robustness] 增强了对 Masks 文件夹的容错处理，没有 Mask 文件也不会影响程序执行。
"""

import os
import re
import glob
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import matplotlib

# 强制使用交互式后端
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rcParams
from matplotlib import gridspec
from scipy.stats import gaussian_kde

# =========================
# 0. 全局配置
# =========================
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'stix',
    "font.size": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlesize": 13,
    "figure.titlesize": 16
}
rcParams.update(config)

# --- 基础路径配置 ---
OUT_DIR = r"H:\InSAR_Mosaic_Project"
MODEL_DIR_BASE = os.path.join(OUT_DIR, "Models")
RESULT_DIR_BASE = os.path.join(OUT_DIR, "Results")
MASK_DIR_BASE = os.path.join(OUT_DIR, "Masks")
QC_DIR_BASE = os.path.join(OUT_DIR, "Quality_Report_Final")

NODATA_VALUE = -9999.0

logging.basicConfig(level=logging.INFO, format="%(message)s")


# =========================
# 1. 辅助工具函数
# =========================

class SinglePolylineDrawer:
    def __init__(self, ax):
        self.ax = ax
        self.pts = []
        self.vis_line, = ax.plot([], [], 'k-', linewidth=2, label='Profile')
        self.vis_pts, = ax.plot([], [], 'y.', markersize=8)
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        print(">>> [交互操作] 1.左键点击画线 | 2.右键点击结束")

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1:
            self.pts.append((event.xdata, event.ydata))
            self.update_plot()
        elif event.button == 3 or event.button == 2:
            if len(self.pts) < 2:
                print(">>> 点数不足，无法生成剖面。")
            else:
                print(">>> 绘制完成，正在基于该剖面线计算多维精度并保存数据...")
                plt.close(self.ax.figure)

    def update_plot(self):
        if self.pts:
            xs, ys = zip(*self.pts)
            self.vis_line.set_data(xs, ys)
            self.vis_pts.set_data(xs, ys)
        self.ax.figure.canvas.draw()


def interpolate_polyline(pts):
    if len(pts) < 2: return np.array([]), np.array([])
    dense_r, dense_c = [], []
    cols = [p[0] for p in pts];
    rows = [p[1] for p in pts]
    for k in range(len(pts) - 1):
        r0, c0 = rows[k], cols[k]
        r1, c1 = rows[k + 1], cols[k + 1]
        dist = np.hypot(r1 - r0, c1 - c0)
        num_steps = int(dist * 2.0)
        if num_steps < 2: num_steps = 2
        rr = np.linspace(r0, r1, num_steps)
        cc = np.linspace(c0, c1, num_steps)
        if k < len(pts) - 2:
            dense_r.extend(rr[:-1]);
            dense_c.extend(cc[:-1])
        else:
            dense_r.extend(rr);
            dense_c.extend(cc)
    return np.array(dense_r), np.array(dense_c)


def manual_draw_single(diff_map, mask, title_info=""):
    fig_click, ax = plt.subplots(figsize=(10, 8))
    valid_vals = diff_map[mask]
    if len(valid_vals) > 0:
        abs_max = np.nanpercentile(np.abs(valid_vals), 99)
        if abs_max < 0.1: abs_max = 1.0
    else:
        abs_max = 1.0

    display_img = np.where(mask, diff_map, np.nan)
    im = ax.imshow(display_img, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, ax=ax, label='Residual (mm/yr)')
    instr = "L-Click: Add Point  |  R-Click: FINISH"
    ax.set_title(f"{title_info}\n{instr}", color='blue', fontweight='bold', fontsize=14)
    drawer = SinglePolylineDrawer(ax)
    plt.show(block=True)
    r, c = interpolate_polyline(drawer.pts)
    return (r, c)


def reproject_to_grid(src, profile):
    arr = np.empty((profile["height"], profile["width"]), dtype=np.float32)
    reproject(source=rasterio.band(src, 1), destination=arr, src_transform=src.transform, src_crs=src.crs,
              dst_transform=profile["transform"], dst_crs=profile["crs"], resampling=Resampling.bilinear,
              dst_nodata=np.nan)
    return arr


def plot_density_scatter(ax, x, y, title):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask];
    y = y[mask]
    if len(x) == 0: return

    if len(x) > 20000:
        idx = np.random.choice(len(x), size=20000, replace=False)
        x_samp, y_samp = x[idx], y[idx]
    else:
        x_samp, y_samp = x, y

    try:
        xy = np.vstack([x_samp, y_samp])
        z = gaussian_kde(xy)(xy)
        idx_sort = z.argsort()
        x_samp, y_samp, z = x_samp[idx_sort], y_samp[idx_sort], z[idx_sort]
    except:
        z = 'blue'

    min_val = min(x_samp.min(), y_samp.min())
    max_val = max(x_samp.max(), y_samp.max())

    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.0, alpha=0.6)
    ax.scatter(x_samp, y_samp, c=z, s=4, cmap='Spectral_r', alpha=0.8, edgecolors='none')

    # 计算 r
    try:
        r_val = np.corrcoef(x_samp, y_samp)[0, 1]
    except:
        r_val = np.nan

    ax.set_title(f"{title}\n[Pearson r = {r_val:.3f}]", fontweight='bold', pad=10, color='darkred')

    ax.set_xlabel("Img 1 (mm/yr)")
    ax.set_ylabel("Img 2 (mm/yr)")
    ax.grid(True, linestyle=':', alpha=0.5)


def calculate_metrics(d_raw, d_corr, v1_raw=None, v2_raw=None, v1_corr=None, v2_corr=None):
    if len(d_raw) == 0:
        return {k: np.nan for k in
                ['rmse_r', 'rmse_c', 'mae_r', 'mae_c', 'std_r', 'std_c', 'med_r', 'med_c',
                 'me_r', 'me_c', 'p2_r', 'p2_c', 'r_raw', 'r_corr', 'imp', 'pts']}

    # 基础绝对指标
    rmse_raw = np.sqrt(np.mean(d_raw ** 2));
    rmse_corr = np.sqrt(np.mean(d_corr ** 2))
    mae_raw = np.mean(np.abs(d_raw));
    mae_corr = np.mean(np.abs(d_corr))
    std_raw = np.std(d_raw);
    std_corr = np.std(d_corr)
    med_raw = np.median(np.abs(d_raw));
    med_corr = np.median(np.abs(d_corr))

    # 新维度 1：Mean Bias (系统性偏差)
    me_raw = np.mean(d_raw)
    me_corr = np.mean(d_corr)

    # 新维度 2：容忍度达标率 (< 2 mm/yr 占比)
    p2_raw = np.sum(np.abs(d_raw) < 2.0) / len(d_raw) * 100
    p2_corr = np.sum(np.abs(d_corr) < 2.0) / len(d_corr) * 100

    # 新维度 3：相关系数 (Pearson r)
    r_raw = np.corrcoef(v1_raw, v2_raw)[0, 1] if (v1_raw is not None and len(v1_raw) > 1) else np.nan
    r_corr = np.corrcoef(v1_corr, v2_corr)[0, 1] if (v1_corr is not None and len(v1_corr) > 1) else np.nan

    imp = (rmse_raw - rmse_corr) / (rmse_raw + 1e-9) * 100

    return {
        'rmse_r': rmse_raw, 'rmse_c': rmse_corr,
        'mae_r': mae_raw, 'mae_c': mae_corr,
        'std_r': std_raw, 'std_c': std_corr,
        'med_r': med_raw, 'med_c': med_corr,
        'me_r': me_raw, 'me_c': me_corr,
        'p2_r': p2_raw, 'p2_c': p2_corr,
        'r_raw': r_raw, 'r_corr': r_corr,
        'imp': imp, 'pts': len(d_raw)
    }


# =========================
# 2. 核心验证类
# =========================
class MosaicValidator:
    def __init__(self, image_list, model_path, mask_dir, label="Task"):
        self.images = image_list
        self.label = label
        self.mask_dir = mask_dir
        self.corrections = None
        self.ref_x = 0.0;
        self.ref_y = 0.0

        # 对外部 Mask 文件夹的无侵入式容错：若没有文件夹或为空，则仅给出提示，不阻断运行
        if mask_dir and not os.path.exists(mask_dir):
            logging.info(f"[{label}] 提示: Mask 目录不存在 ({mask_dir})，将完全依赖动态有效值掩膜。")

        if os.path.exists(model_path):
            self.corrections = np.load(model_path)
            logging.info(f"[{label}] Model loaded: {model_path}")
        else:
            logging.error(f"[{label}] Model NOT found: {model_path}")
            raise FileNotFoundError

        xs, ys = [], []
        for img in self.images:
            with rasterio.open(img['vel']) as src:
                l, b, r, t = src.bounds
                xs.append((l + r) / 2);
                ys.append((t + b) / 2)
        self.ref_x = np.mean(xs);
        self.ref_y = np.mean(ys)

    def run_validation(self, img_save_dir, current_qc_dir):
        if not os.path.exists(img_save_dir): os.makedirs(img_save_dir)

        # 准备写入包含全维度指标的 CSV
        stats_csv_path = os.path.join(current_qc_dir, f"{self.label}_Stats_Final.csv")
        header = "Pair,Name1,Name2,Points,RMSE_Raw,RMSE_Cor,MAE_Raw,MAE_Cor,STD_Raw,STD_Cor,MED_Raw,MED_Cor,ME_Raw,ME_Cor,P<2mm_Raw,P<2mm_Cor,R_Raw,R_Cor,Imp_%,Status\n"
        if not os.path.exists(stats_csv_path):
            with open(stats_csv_path, 'w') as f: f.write(header)

        logging.info(f"--- Processing Group: {self.label} ---")
        for i in range(len(self.images)):
            for j in range(i + 1, len(self.images)):
                self._process_pair(i, j, img_save_dir, stats_csv_path)

    def _process_pair(self, i, j, save_dir, stats_csv_file):
        img1, img2 = self.images[i], self.images[j]
        name1 = os.path.basename(img1['vel']).rsplit('.', 1)[0]
        name2 = os.path.basename(img2['vel']).rsplit('.', 1)[0]

        # 1. 重采样
        with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2:
            xl = max(src1.bounds.left, src2.bounds.left)
            xr = min(src1.bounds.right, src2.bounds.right)
            yb = max(src1.bounds.bottom, src2.bounds.bottom)
            yt = min(src1.bounds.top, src2.bounds.top)
            if xr <= xl or yt <= yb: return
            width = int((xr - xl) / src1.res[0]);
            height = int((yt - yb) / src1.res[1])
            if width < 10 or height < 10: return
            trans = from_origin(xl, yt, src1.res[0], src1.res[1])
            prof = {'height': height, 'width': width, 'transform': trans, 'crs': src1.crs}
            v1_m = reproject_to_grid(src1, prof)
            v2_m = reproject_to_grid(src2, prof)

        # 2. 计算校正曲面
        rows, cols = np.indices(v1_m.shape)
        px, py = trans * (cols, rows)
        px -= self.ref_x;
        py -= self.ref_y

        def poly(p, x, y):
            return p[0] + p[1] * x + p[2] * y + p[3] * x ** 2 + p[4] * y ** 2 + p[5] * x * y

        c1_m = poly(self.corrections[i], px, py)
        c2_m = poly(self.corrections[j], px, py)

        SCALE = 1000.0
        v1_mm = v1_m * SCALE;
        v2_mm = v2_m * SCALE
        surf_mm = (c1_m - c2_m) * SCALE
        v1_corr_mm = (v1_m - c1_m) * SCALE
        v2_corr_mm = (v2_m - c2_m) * SCALE
        diff_raw_mm = v1_mm - v2_mm
        diff_corr_mm = v1_corr_mm - v2_corr_mm

        # 3. 基础有效掩膜 (用于作为交互底图) - 这里主要依据数据本身质量动态生成
        mask_basic = np.isfinite(diff_raw_mm) & np.isfinite(diff_corr_mm) & (v1_m != NODATA_VALUE) & (
                    v2_m != NODATA_VALUE)
        if np.sum(mask_basic) < 300: return

        # 4. 交互绘图 (画出剖面线)
        line_r, line_c = manual_draw_single(diff_corr_mm, mask_basic, title_info=f"{name1} vs {name2}")

        # 5. 提取剖面线上面的数据，并根据线数据计算所有精度
        if len(line_r) > 1:
            idx_r = np.clip(np.round(line_r).astype(int), 0, v1_corr_mm.shape[0] - 1)
            idx_c = np.clip(np.round(line_c).astype(int), 0, v1_corr_mm.shape[1] - 1)

            p_x = np.cumsum(np.insert(np.sqrt(np.diff(line_r) ** 2 + np.diff(line_c) ** 2), 0, 0))

            p_v1_raw = v1_mm[idx_r, idx_c]
            p_v2_raw = v2_mm[idx_r, idx_c]
            p_v1_corr = v1_corr_mm[idx_r, idx_c]
            p_v2_corr = v2_corr_mm[idx_r, idx_c]

            p_diff_raw = diff_raw_mm[idx_r, idx_c]
            p_diff_corr = diff_corr_mm[idx_r, idx_c]

            valid = np.isfinite(p_diff_raw) & np.isfinite(p_diff_corr)
            p_v1_raw = p_v1_raw[valid]
            p_v2_raw = p_v2_raw[valid]
            p_v1_corr = p_v1_corr[valid]
            p_v2_corr = p_v2_corr[valid]
            p_diff_raw = p_diff_raw[valid]
            p_diff_corr = p_diff_corr[valid]

            metrics_profile = calculate_metrics(p_diff_raw, p_diff_corr, p_v1_raw, p_v2_raw, p_v1_corr, p_v2_corr)
        else:
            p_x = np.array([])
            p_v1_raw = p_v2_raw = p_v1_corr = p_v2_corr = p_diff_raw = p_diff_corr = np.array([])
            metrics_profile = calculate_metrics(np.array([]), np.array([]))

        # 6. 将这段线的多维专属精度写入 CSV 中
        line_csv = f"{i}-{j},{name1},{name2},{metrics_profile['pts']},{metrics_profile['rmse_r']:.3f},{metrics_profile['rmse_c']:.3f},{metrics_profile['mae_r']:.3f},{metrics_profile['mae_c']:.3f},{metrics_profile['std_r']:.3f},{metrics_profile['std_c']:.3f},{metrics_profile['med_r']:.3f},{metrics_profile['med_c']:.3f},{metrics_profile['me_r']:.3f},{metrics_profile['me_c']:.3f},{metrics_profile['p2_r']:.1f},{metrics_profile['p2_c']:.1f},{metrics_profile['r_raw']:.3f},{metrics_profile['r_corr']:.3f},{metrics_profile['imp']:.1f},ProfileData\n"
        with open(stats_csv_file, 'a') as f:
            f.write(line_csv)

        # ==========================================
        # 7. (修改后) 将这条线上的每个点对点的原始和校正数据完整导出
        # ==========================================
        if len(line_r) > 1:
            profile_csv_name = f"ProfileData_{i}_{j}.csv"
            profile_csv_path = os.path.join(save_dir, profile_csv_name)

            with open(profile_csv_path, 'w') as f:
                f.write("Distance_px,Row_Idx,Col_Idx,Vel_Left_Raw_mm,Vel_Right_Raw_mm,Vel_Left_Cor_mm,Vel_Right_Cor_mm,Res_Raw_mm,Res_Cor_mm\n")
                for dist, r, c, v1_raw, v2_raw, v1_cor, v2_cor, diff_raw, diff_cor in zip(
                        p_x, idx_r, idx_c, p_v1_raw, p_v2_raw, p_v1_corr, p_v2_corr, p_diff_raw, p_diff_corr):
                    f.write(f"{dist:.3f},{r},{c},{v1_raw:.3f},{v2_raw:.3f},{v1_cor:.3f},{v2_cor:.3f},{diff_raw:.3f},{diff_cor:.3f}\n")

        # 8. 存图
        self._plot_visual_page(name1, name2, v1_corr_mm, v2_corr_mm, diff_corr_mm, mask_basic,
                               line_r, line_c, save_dir, i, j)

        self._plot_stats_page(name1, name2, v1_corr_mm, v2_corr_mm, surf_mm, mask_basic,
                              p_v1_raw, p_v2_raw, p_v1_corr, p_v2_corr,
                              p_diff_raw, p_diff_corr, line_r, line_c, metrics_profile,
                              os.path.join(save_dir, f"Check_{i}_{j}_2_Stats_Profile.png"), "Profile Line Only")

    def _plot_visual_page(self, name1, name2, img1, img2, diff_map, mask_show, line_r, line_c, out_dir, idx1, idx2):
        fig1 = plt.figure(figsize=(22, 5.5))
        gs1 = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.15)

        all_vis_data = np.concatenate([img1[mask_show], img2[mask_show], diff_map[mask_show]])
        if len(all_vis_data) > 0:
            global_max_abs = np.percentile(np.abs(all_vis_data), 99)
            if global_max_abs < 0.1: global_max_abs = 1.0
        else:
            global_max_abs = 5.0
        global_norm = Normalize(vmin=-global_max_abs, vmax=global_max_abs)

        ax1 = fig1.add_subplot(gs1[0, 0])
        im1 = ax1.imshow(np.where(mask_show, img1, np.nan), cmap='RdBu_r', norm=global_norm)
        ax1.set_title(f"(a) Left: {name1}\n(Corrected, mm/yr)", fontweight='bold')
        ax1.axis('off')

        ax2 = fig1.add_subplot(gs1[0, 1])
        im2 = ax2.imshow(np.where(mask_show, img2, np.nan), cmap='RdBu_r', norm=global_norm)
        ax2.set_title(f"(b) Right: {name2}\n(Corrected, mm/yr)", fontweight='bold')
        ax2.axis('off')

        ax3 = fig1.add_subplot(gs1[0, 2])
        im3 = ax3.imshow(np.where(mask_show, diff_map, np.nan), cmap='RdBu_r', norm=global_norm)
        ax3.set_title(f"(c) Residuals\n(Unified Scale, mm/yr)", fontweight='bold')
        ax3.axis('off')

        ax4 = fig1.add_subplot(gs1[0, 3])
        ax4.imshow(np.where(mask_show, diff_map, np.nan), cmap='RdBu_r', norm=global_norm)
        if len(line_r) > 1: ax4.plot(line_c, line_r, 'k-', linewidth=2.5, alpha=0.8)
        ax4.set_title(f"(d) Profile Location", fontweight='bold')
        ax4.axis('off')

        cax_ab = fig1.add_axes([0.15, 0.08, 0.28, 0.025])
        plt.colorbar(im1, cax=cax_ab, orientation='horizontal', label='Velocity (mm/yr)')
        cax_cd = fig1.add_subplot(gs1[0, 4])
        plt.colorbar(im3, cax=cax_cd, label='Residual (mm/yr)')

        plt.suptitle(f"Visual Report: {name1} vs {name2} (Unified Scale)", fontsize=16, fontweight='bold', y=0.98)
        fig1.savefig(os.path.join(out_dir, f"Check_{idx1}_{idx2}_1_Visual.png"), dpi=600, bbox_inches='tight')
        plt.close(fig1)

    def _plot_stats_page(self, name1, name2, img1, img2, surf_map, mask_2d,
                         raw_x, raw_y, corr_x, corr_y, d_raw, d_corr,
                         line_r, line_c, metrics, save_path, mode_label):
        fig2 = plt.figure(figsize=(16, 12))
        gs2 = fig2.add_gridspec(3, 2, hspace=0.45, wspace=0.25)

        def smart_label(n):
            return n[:10] + "..." + n[-8:] if len(n) > 22 else n

        ax_e = fig2.add_subplot(gs2[0, 0])
        if len(line_r) > 1:
            idx_r = np.clip(np.round(line_r).astype(int), 0, img1.shape[0] - 1)
            idx_c = np.clip(np.round(line_c).astype(int), 0, img1.shape[1] - 1)
            p_x = np.cumsum(np.insert(np.sqrt(np.diff(line_r) ** 2 + np.diff(line_c) ** 2), 0, 0))
            ax_e.plot(p_x, img1[idx_r, idx_c], 'r-', linewidth=1.5, alpha=0.8, label=smart_label(name1))
            ax_e.plot(p_x, img2[idx_r, idx_c], 'b--', linewidth=1.5, alpha=0.8, label=smart_label(name2))
            ax_e.set_title("(e) Velocity Profile Comparison", fontweight='bold')
            ax_e.set_xlabel("Distance (pixels)");
            ax_e.set_ylabel("Velocity (mm/yr)")
            ax_e.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=10, ncol=2, frameon=True)
            ax_e.grid(True, linestyle=':', alpha=0.5)
        else:
            ax_e.text(0.5, 0.5, "No Profile Drawn", ha='center')

        ax_f = fig2.add_subplot(gs2[0, 1])
        try:
            combine_d = np.concatenate([d_raw, d_corr])
            if len(combine_d) > 0:
                bin_min, bin_max = np.percentile(combine_d, [0.1, 99.9])
                bins = np.linspace(bin_min, bin_max, 60)
                ax_f.hist(d_raw, bins=bins, color='gray', alpha=0.5, label='Original', density=True, edgecolor='none')
                ax_f.hist(d_corr, bins=bins, color='red', alpha=0.5, label='Corrected', density=True, edgecolor='none')
                ax_f.hist(d_raw, bins=bins, color='black', histtype='step', linewidth=1, density=True, alpha=0.3)
                ax_f.hist(d_corr, bins=bins, color='red', histtype='step', linewidth=1, density=True, alpha=0.5)
                ax_f.axvline(0, color='green', linestyle=':', linewidth=1.5, label='Zero')
                ax_f.set_title("(f) Profile Residual Histogram", fontweight='bold')
                ax_f.set_xlabel("Residual (mm/yr)");
                ax_f.set_ylabel("Density (Normalized)")
                ax_f.set_xlim(bin_min, bin_max)
            else:
                ax_f.text(0.5, 0.5, "No Data", ha='center')
            ax_f.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=10, ncol=3, frameon=True)
            ax_f.grid(True, linestyle=':', alpha=0.3)
        except Exception as e:
            ax_f.text(0.5, 0.5, "Error Plotting Hist", ha='center')

        ax_sc1 = fig2.add_subplot(gs2[1, 0])
        plot_density_scatter(ax_sc1, raw_x, raw_y, "(g) Profile Scatter: Original")
        ax_sc2 = fig2.add_subplot(gs2[1, 1])
        plot_density_scatter(ax_sc2, corr_x, corr_y, "(h) Profile Scatter: Corrected")

        ax_i = fig2.add_subplot(gs2[2, 0])
        v_surf_max = np.nanmax(np.abs(surf_map[mask_2d])) if np.sum(mask_2d) > 0 else 0.5
        im_surf = ax_i.imshow(np.where(mask_2d, surf_map, np.nan), cmap='RdBu_r', vmin=-v_surf_max, vmax=v_surf_max)
        ax_i.set_title("(i) Estimated Correction Surface", fontweight='bold')
        ax_i.axis('off')
        plt.colorbar(im_surf, ax=ax_i, fraction=0.046, pad=0.04, label='mm/yr')

        ax_sum = fig2.add_subplot(gs2[2, 1]);
        ax_sum.axis('off')
        bg_color = "#ffe4e1" if "Profile" in mode_label else "#ffffe0"

        txt = (
            f"PROFILE STATISTICAL REPORT (mm/yr)\n"
            f"=========================================\n"
            f"Left     : {smart_label(name1)}\n"
            f"Right    : {smart_label(name2)}\n"
            f"Mode     : {mode_label} | Pts: {metrics['pts']}\n"
            f"-----------------------------------------\n"
            f"METRIC   |  ORIGINAL  |  CORRECTED \n"
            f"-----------------------------------------\n"
            f"RMSE     |  {metrics['rmse_r']:8.3f}  |  {metrics['rmse_c']:8.3f}\n"
            f"MAE      |  {metrics['mae_r']:8.3f}  |  {metrics['mae_c']:8.3f}\n"
            f"STD      |  {metrics['std_r']:8.3f}  |  {metrics['std_c']:8.3f}\n"
            f"MED(Abs) |  {metrics['med_r']:8.3f}  |  {metrics['med_c']:8.3f}\n"
            f"Mean Bias|  {metrics['me_r']:8.3f}  |  {metrics['me_c']:8.3f}\n"
            f"< 2 mm/yr|  {metrics['p2_r']:7.1f}% |  {metrics['p2_c']:7.1f}%\n"
            f"Pearson r|  {metrics['r_raw']:8.3f}  |  {metrics['r_corr']:8.3f}\n"
            f"-----------------------------------------\n"
            f"Improvement (RMSE): {metrics['imp']:.1f} %\n"
            f"========================================="
        )
        ax_sum.text(0.5, 0.5, txt, fontsize=10.5, family='monospace', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=1", facecolor=bg_color, edgecolor="black"))

        plt.suptitle(f"Statistical Report: {name1} vs {name2}\n({mode_label})", fontsize=16, fontweight='bold', y=0.96)
        fig2.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig2)


if __name__ == "__main__":
    # ==========================================
    # 1. 配置输入：在这里手动修改目标子文件夹名称
    # ==========================================
    target_year = 19
    sub_dir_name = "19_1"  # <--- 每次运行前修改

    current_model_dir = os.path.join(MODEL_DIR_BASE, sub_dir_name)
    current_mask_dir = os.path.join(MASK_DIR_BASE, sub_dir_name)
    current_result_dir = os.path.join(RESULT_DIR_BASE, sub_dir_name)
    current_qc_dir = os.path.join(QC_DIR_BASE, sub_dir_name)

    # 检查模型文件夹是否存在
    if not os.path.exists(current_model_dir):
        print(f"[-] Error: 找不到指定的模型文件夹路径: {current_model_dir}")
        input(">>> 按回车键退出程序...")  # 阻止窗口闪退
        exit()

    if not os.path.exists(current_qc_dir): os.makedirs(current_qc_dir)

    # 增强鲁棒性：如果 Masks 文件夹缺失，主动创建占位目录
    if not os.path.exists(current_mask_dir):
        os.makedirs(current_mask_dir, exist_ok=True)
        print(f"[*] Info: {current_mask_dir} 不存在，已自动创建占位文件夹。")

    print(f"=== STARTING V36 (Multi-Dimensional Strict Profile Metrics: {sub_dir_name}) ===")
    print(f"{'=' * 50}")

    # ==========================================
    # 2. 影像文件列表配置
    # ==========================================
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
    tracks = {}
    pattern = re.compile(r'vel_\d+_(\d+)_\d+')
    for item in inputs:
        filename = os.path.basename(item['vel'])
        match = pattern.search(filename)
        path_id = match.group(1) if match else "Unknown"
        if path_id not in tracks: tracks[path_id] = []
        tracks[path_id].append(item)

    # 全局模型路径
    global_model_path = os.path.join(current_model_dir, "Global_Model.npy")
    processed_count = 0

    # --- 3. 处理同轨(Track)的拼接验证 ---
    for pid, imgs in tracks.items():
        if len(imgs) < 2:
            print(f"[-] Track {pid} 影像数量不足 2 景，跳过。")
            continue

        # 兼容逻辑：优先找全局模型，如果没有，再找分轨模型
        track_model_path = os.path.join(current_model_dir, f"Track_{pid}_model.npy")
        model_to_use = None

        if os.path.exists(global_model_path):
            model_to_use = global_model_path
        elif os.path.exists(track_model_path):
            model_to_use = track_model_path

        if model_to_use:
            try:
                print(f"[*] 正在使用模型: {os.path.basename(model_to_use)} 处理 Track {pid}")
                val = MosaicValidator(imgs, model_to_use, mask_dir=current_mask_dir, label=f"Track_{pid}")
                img_save_dir = os.path.join(current_qc_dir, f"Track_{pid}")
                val.run_validation(img_save_dir, current_qc_dir)
                processed_count += 1
            except Exception as e:
                print(f"[!] Error Track {pid} in {sub_dir_name}: {e}")
        else:
            print(f"[-] 找不到 Track {pid} 对应的模型文件 (Global 或 分轨均缺失)，已跳过。")

    # --- 4. 处理跨轨(CrossTrack)的拼接验证 ---
    if os.path.exists(current_result_dir):
        strip_vels = glob.glob(os.path.join(current_result_dir, "Strip_*_vel.tif"))
        strips = []
        for v_path in strip_vels:
            c_path = v_path.replace("_vel.tif", "_coh.tif")
            if os.path.exists(c_path): strips.append({'vel': v_path, 'coh': c_path})
        strips.sort(key=lambda x: x['vel'])

        # 兼容逻辑：优先找全局模型，如果没有，再找旧版跨轨模型
        cross_model_path = os.path.join(current_model_dir, "CrossTrack_model.npy")
        model_to_use = None

        if os.path.exists(global_model_path):
            model_to_use = global_model_path
        elif os.path.exists(cross_model_path):
            model_to_use = cross_model_path

        if len(strips) >= 2 and model_to_use:
            try:
                print(f"[*] 正在使用模型: {os.path.basename(model_to_use)} 处理 CrossTrack")
                val = MosaicValidator(strips, model_to_use, mask_dir=current_mask_dir, label="CrossTrack")
                img_save_dir = os.path.join(current_qc_dir, "CrossTrack")
                val.run_validation(img_save_dir, current_qc_dir)
                processed_count += 1
            except Exception as e:
                print(f"[!] Error CrossTrack in {sub_dir_name}: {e}")
        else:
            if not model_to_use:
                print(f"[-] 找不到 CrossTrack 对应的模型文件 (Global 或 跨轨专属均缺失)，已跳过跨轨验证")
            else:
                print(f"[-] 跨轨验证跳过：Strips 数量不足 ({len(strips)} 景)")
    else:
        print(f"[-] 找不到结果目录: {current_result_dir}，跳过跨轨验证。")

    print(f"\nAll Done. 任务结束，共处理了 {processed_count} 个验证组。")
    input(">>> 按回车键关闭窗口...")