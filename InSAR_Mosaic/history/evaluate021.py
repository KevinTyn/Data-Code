# -*- coding: utf-8 -*-
"""
Script 2 (Final V28 - Unified Scale & Hist): InSAR Mosaic Validator
功能：读取模型和掩膜，进行单剖面交互验证。
特性：
1. [Layout] 图例全部移至绘图区域外部（下方）。
2. [Metrics] 全指标统计 (RMSE, MAE, STD, MED)。
3. [Visual] 图a/b/c/d 统一色标范围 (Global Scale)，方便量级对比。
4. [Visual] 图f 改为直方图 (Histogram) 对比。
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

# --- 路径配置 (请保持与之前一致) ---
OUT_DIR = r"H:\\InSAR_Mosaic_Project"
MODEL_DIR = os.path.join(OUT_DIR, "Models")
RESULT_DIR = os.path.join(OUT_DIR, "Results")
MASK_DIR = os.path.join(OUT_DIR, "Masks")
QC_DIR = os.path.join(OUT_DIR, "Quality_Report_Final")

NODATA_VALUE = -9999.0

if not os.path.exists(QC_DIR): os.makedirs(QC_DIR)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# =========================
# 1. 辅助工具函数
# =========================

class SinglePolylineDrawer:
    """[交互] 左键加点，右键结束"""

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
                print(">>> 绘制完成。")
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
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Img 1 (mm/yr)")
    ax.set_ylabel("Img 2 (mm/yr)")
    ax.grid(True, linestyle=':', alpha=0.5)


# =========================
# 2. 核心验证类
# =========================
class MosaicValidator:
    def __init__(self, image_list, model_path, label="Task"):
        self.images = image_list
        self.label = label
        self.corrections = None
        self.ref_x = 0.0;
        self.ref_y = 0.0

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

    def run_validation(self, output_dir):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        csv_path = os.path.join(output_dir, f"{self.label}_Stats_Final.csv")

        header = "Pair,Name1,Name2,Points,RMSE_Raw,RMSE_Cor,MAE_Raw,MAE_Cor,STD_Raw,STD_Cor,MED_Raw,MED_Cor,Imp_%,Status\n"
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f: f.write(header)

        logging.info(f"--- Processing Group: {self.label} ---")
        for i in range(len(self.images)):
            for j in range(i + 1, len(self.images)):
                self._process_pair(i, j, output_dir, csv_path)

    def _process_pair(self, i, j, save_dir, csv_file):
        img1, img2 = self.images[i], self.images[j]
        name1 = os.path.basename(img1['vel']).rsplit('.', 1)[0]
        name2 = os.path.basename(img2['vel']).rsplit('.', 1)[0]

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

        rows, cols = np.indices(v1_m.shape)
        px, py = trans * (cols, rows)
        px -= self.ref_x;
        py -= self.ref_y

        def poly(p, x, y):
            return p[0] + p[1] * x + p[2] * y + p[3] * x ** 2 + p[4] * y ** 2 + p[5] * x * y

        c1_m = poly(self.corrections[i], px, py)
        c2_m = poly(self.corrections[j], px, py)

        SCALE = 1000.0
        v1_mm = v1_m * SCALE
        v2_mm = v2_m * SCALE
        surf_mm = (c1_m - c2_m) * SCALE
        v1_corr_mm = (v1_m - c1_m) * SCALE
        v2_corr_mm = (v2_m - c2_m) * SCALE
        diff_raw_mm = v1_mm - v2_mm
        diff_corr_mm = v1_corr_mm - v2_corr_mm

        mask_basic = np.isfinite(diff_raw_mm) & np.isfinite(diff_corr_mm) & (v1_m != NODATA_VALUE) & (
                    v2_m != NODATA_VALUE)
        if np.sum(mask_basic) < 300: return

        mask_file = os.path.join(MASK_DIR, f"Mask_{i}_{j}.npy")
        mask_final = None;
        status_tag = "Robust"
        if os.path.exists(mask_file):
            try:
                packed = np.load(mask_file)
                if packed.size * 8 >= v1_mm.size:
                    loaded_mask = np.unpackbits(packed, count=v1_mm.size).reshape(v1_mm.shape).astype(bool)
                    mask_final = mask_basic & loaded_mask
                    status_tag = "Exact"
            except:
                pass

        if mask_final is None:
            resid = diff_corr_mm[mask_basic]
            mask_final = mask_basic & (np.abs(diff_corr_mm - np.median(resid)) < 3 * np.std(resid))

        d_raw = diff_raw_mm[mask_final]
        d_corr = diff_corr_mm[mask_final]

        rmse_raw = np.sqrt(np.mean(d_raw ** 2));
        rmse_corr = np.sqrt(np.mean(d_corr ** 2))
        mae_raw = np.mean(np.abs(d_raw));
        mae_corr = np.mean(np.abs(d_corr))
        std_raw = np.std(d_raw);
        std_corr = np.std(d_corr)
        med_raw = np.median(np.abs(d_raw));
        med_corr = np.median(np.abs(d_corr))
        imp = (rmse_raw - rmse_corr) / (rmse_raw + 1e-9) * 100

        line = f"{i}-{j},{name1},{name2},{np.sum(mask_final)},{rmse_raw:.3f},{rmse_corr:.3f},{mae_raw:.3f},{mae_corr:.3f},{std_raw:.3f},{std_corr:.3f},{med_raw:.3f},{med_corr:.3f},{imp:.1f},{status_tag}\n"
        with open(csv_file, 'a') as f:
            f.write(line)

        line_r, line_c = manual_draw_single(diff_corr_mm, mask_basic, title_info=f"{name1} vs {name2}")

        metrics = {
            'rmse_r': rmse_raw, 'rmse_c': rmse_corr,
            'mae_r': mae_raw, 'mae_c': mae_corr,
            'std_r': std_raw, 'std_c': std_corr,
            'med_r': med_raw, 'med_c': med_corr,
            'imp': imp, 'pts': np.sum(mask_final)
        }

        self._plot_dual_report(name1, name2, v1_corr_mm, v2_corr_mm, diff_corr_mm, surf_mm, mask_basic,
                               v1_mm[mask_final], v2_mm[mask_final], v1_corr_mm[mask_final], v2_corr_mm[mask_final],
                               d_raw, d_corr, metrics, save_dir, line_r, line_c, status_tag, idx1=i, idx2=j)

    def _plot_dual_report(self, name1, name2, img1, img2, diff_map, surf_map, mask_show,
                          raw_x, raw_y, corr_x, corr_y, d_raw, d_corr,
                          metrics, out_dir, line_r, line_c, status_tag, idx1, idx2):

        # =================================================
        # FIGURE 1: Visual Inspection (Unified Scale)
        # =================================================
        fig1 = plt.figure(figsize=(22, 5.5))
        gs1 = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.15)

        # -------------------------------------------------------------------------
        # [修改点 1] 统一色标：计算 img1, img2, diff_map 的全局最大绝对值
        # -------------------------------------------------------------------------
        # 将三个图的数据合并，找出99分位的绝对值最大值，用于统一 colorbar
        all_vis_data = np.concatenate([img1[mask_show], img2[mask_show], diff_map[mask_show]])
        if len(all_vis_data) > 0:
            global_max_abs = np.percentile(np.abs(all_vis_data), 99)
            if global_max_abs < 0.1: global_max_abs = 1.0
        else:
            global_max_abs = 5.0

        # 统一定义 Norm
        global_norm = Normalize(vmin=-global_max_abs, vmax=global_max_abs)

        # (a) Left
        ax1 = fig1.add_subplot(gs1[0, 0])
        im1 = ax1.imshow(np.where(mask_show, img1, np.nan), cmap='RdBu_r', norm=global_norm)
        ax1.set_title(f"(a) Left: {name1}\n(Corrected, mm/yr)", fontweight='bold')
        ax1.axis('off')

        # (b) Right
        ax2 = fig1.add_subplot(gs1[0, 1])
        im2 = ax2.imshow(np.where(mask_show, img2, np.nan), cmap='RdBu_r', norm=global_norm)
        ax2.set_title(f"(b) Right: {name2}\n(Corrected, mm/yr)", fontweight='bold')
        ax2.axis('off')

        # (c) Residuals (Scale Unified with a/b)
        ax3 = fig1.add_subplot(gs1[0, 2])
        im3 = ax3.imshow(np.where(mask_show, diff_map, np.nan), cmap='RdBu_r', norm=global_norm)
        ax3.set_title(f"(c) Residuals\n(Unified Scale, mm/yr)", fontweight='bold')
        ax3.axis('off')

        # (d) Profile Loc (Scale Unified)
        ax4 = fig1.add_subplot(gs1[0, 3])
        ax4.imshow(np.where(mask_show, diff_map, np.nan), cmap='RdBu_r', norm=global_norm)
        if len(line_r) > 1:
            ax4.plot(line_c, line_r, 'k-', linewidth=2.5, alpha=0.8)
        ax4.set_title(f"(d) Profile Location", fontweight='bold')
        ax4.axis('off')

        # Colorbars (由于Scale统一，实际上两个colorbar显示范围是一样的)
        cax_ab = fig1.add_axes([0.15, 0.08, 0.28, 0.025])
        plt.colorbar(im1, cax=cax_ab, orientation='horizontal', label='Velocity (mm/yr)')

        cax_cd = fig1.add_subplot(gs1[0, 4])
        plt.colorbar(im3, cax=cax_cd, label='Residual (mm/yr)')

        plt.suptitle(f"Visual Report: {name1} vs {name2} (Unified Scale)", fontsize=16, fontweight='bold', y=0.98)
        fig1.savefig(os.path.join(out_dir, f"Check_{idx1}_{idx2}_1_Visual.png"), dpi=600, bbox_inches='tight')
        plt.close(fig1)

        # =================================================
        # FIGURE 2: Statistical Analysis
        # =================================================
        fig2 = plt.figure(figsize=(16, 12))
        gs2 = fig2.add_gridspec(3, 2, hspace=0.45, wspace=0.25)

        # (e) Velocity Profile
        ax_e = fig2.add_subplot(gs2[0, 0])

        def smart_label(n):
            if len(n) > 22: return n[:10] + "..." + n[-8:]
            return n

        if len(line_r) > 1:
            idx_r = np.clip(np.round(line_r).astype(int), 0, img1.shape[0] - 1)
            idx_c = np.clip(np.round(line_c).astype(int), 0, img1.shape[1] - 1)
            p_y1 = img1[idx_r, idx_c]
            p_y2 = img2[idx_r, idx_c]
            p_x = np.cumsum(np.insert(np.sqrt(np.diff(line_r) ** 2 + np.diff(line_c) ** 2), 0, 0))

            ax_e.plot(p_x, p_y1, 'r-', linewidth=1.5, alpha=0.8, label=smart_label(name1))
            ax_e.plot(p_x, p_y2, 'b--', linewidth=1.5, alpha=0.8, label=smart_label(name2))
            ax_e.set_title("(e) Velocity Profile Comparison", fontweight='bold')
            ax_e.set_xlabel("Distance (pixels)")
            ax_e.set_ylabel("Velocity (mm/yr)")
            ax_e.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=10, ncol=2, frameon=True)
            ax_e.grid(True, linestyle=':', alpha=0.5)
        else:
            ax_e.text(0.5, 0.5, "No Profile Drawn", ha='center')

        # -------------------------------------------------------------------------
        # [修改点 2] 图(f) 改为直方图 (Histogram)
        # -------------------------------------------------------------------------
        ax_f = fig2.add_subplot(gs2[0, 1])
        try:
            # 确定统一的 Bins 范围
            combine_d = np.concatenate([d_raw, d_corr])
            if len(combine_d) > 0:
                bin_min, bin_max = np.percentile(combine_d, [0.1, 99.9])
                bins = np.linspace(bin_min, bin_max, 60)  # 60个区间

                # 绘制直方图 (density=True表示归一化为概率密度，方便对比形状)
                ax_f.hist(d_raw, bins=bins, color='gray', alpha=0.5, label='Original', density=True, edgecolor='none')
                ax_f.hist(d_corr, bins=bins, color='red', alpha=0.5, label='Corrected', density=True, edgecolor='none')

                # 绘制轮廓线让图更清晰
                ax_f.hist(d_raw, bins=bins, color='black', histtype='step', linewidth=1, density=True, alpha=0.3)
                ax_f.hist(d_corr, bins=bins, color='red', histtype='step', linewidth=1, density=True, alpha=0.5)

                ax_f.axvline(0, color='green', linestyle=':', linewidth=1.5, label='Zero')
                ax_f.set_title("(f) Residual Histogram", fontweight='bold')
                ax_f.set_xlabel("Residual (mm/yr)")
                ax_f.set_ylabel("Density / Count (Normalized)")
                ax_f.set_xlim(bin_min, bin_max)
            else:
                ax_f.text(0.5, 0.5, "No Data", ha='center')

            # 图例仍在外部
            ax_f.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=10, ncol=3, frameon=True)
            ax_f.grid(True, linestyle=':', alpha=0.3)
        except Exception as e:
            print(e)
            ax_f.text(0.5, 0.5, "Error Plotting Hist", ha='center')

        # (g)(h) Scatters
        ax_sc1 = fig2.add_subplot(gs2[1, 0])
        plot_density_scatter(ax_sc1, raw_x, raw_y, "(g) Scatter: Original")
        ax_sc2 = fig2.add_subplot(gs2[1, 1])
        plot_density_scatter(ax_sc2, corr_x, corr_y, "(h) Scatter: Corrected")

        # (i) Correction Surface
        ax_i = fig2.add_subplot(gs2[2, 0])
        v_surf_max = np.nanmax(np.abs(surf_map[mask_show])) if np.sum(mask_show) > 0 else 0.5
        im_surf = ax_i.imshow(np.where(mask_show, surf_map, np.nan), cmap='RdBu_r', vmin=-v_surf_max, vmax=v_surf_max)
        ax_i.set_title("(i) Estimated Correction Surface", fontweight='bold')
        ax_i.axis('off')
        plt.colorbar(im_surf, ax=ax_i, fraction=0.046, pad=0.04, label='mm/yr')

        # (j) Summary Table
        ax_sum = fig2.add_subplot(gs2[2, 1]);
        ax_sum.axis('off')
        bg_color = "#e0ffe0" if "Exact" in status_tag else "#ffffe0"
        txt = (
            f"FULL STATISTICAL REPORT (mm/yr)\n"
            f"=========================================\n"
            f"Left     : {smart_label(name1)}\n"
            f"Right    : {smart_label(name2)}\n"
            f"Mode     : {status_tag} Mask | Pts: {metrics['pts']}\n"
            f"-----------------------------------------\n"
            f"METRIC   |  ORIGINAL  |  CORRECTED \n"
            f"-----------------------------------------\n"
            f"RMSE     |  {metrics['rmse_r']:.3f}     |  {metrics['rmse_c']:.3f}\n"
            f"MAE      |  {metrics['mae_r']:.3f}     |  {metrics['mae_c']:.3f}\n"
            f"STD      |  {metrics['std_r']:.3f}     |  {metrics['std_c']:.3f}\n"
            f"MED(Abs) |  {metrics['med_r']:.3f}     |  {metrics['med_c']:.3f}\n"
            f"-----------------------------------------\n"
            f"Improvement: {metrics['imp']:.1f} %\n"
            f"========================================="
        )
        ax_sum.text(0.5, 0.5, txt, fontsize=10.5, family='monospace', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=1", facecolor=bg_color, edgecolor="black"))

        plt.suptitle(f"Statistical Report: {name1} vs {name2}", fontsize=18, fontweight='bold', y=0.95)
        fig2.savefig(os.path.join(out_dir, f"Check_{idx1}_{idx2}_2_Stats.png"), dpi=600, bbox_inches='tight')
        plt.close(fig2)


if __name__ == "__main__":
    # 1. 配置输入
    target_year = 17

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

    print("=== STARTING V28 (Unified Scale & Histogram) ===")

    for pid, imgs in tracks.items():
        if len(imgs) < 2: continue
        model_path = os.path.join(MODEL_DIR, f"Track_{pid}_model.npy")
        if os.path.exists(model_path):
            try:
                val = MosaicValidator(imgs, model_path, label=f"Track_{pid}")
                val.run_validation(os.path.join(QC_DIR, f"Track_{pid}"))
            except Exception as e:
                print(f"Error Track {pid}: {e}")

    strip_vels = glob.glob(os.path.join(RESULT_DIR, "Strip_*_vel.tif"))
    strips = []
    for v_path in strip_vels:
        c_path = v_path.replace("_vel.tif", "_coh.tif")
        if os.path.exists(c_path): strips.append({'vel': v_path, 'coh': c_path})
    strips.sort(key=lambda x: x['vel'])

    cross_model_path = os.path.join(MODEL_DIR, "CrossTrack_model.npy")
    if len(strips) >= 2 and os.path.exists(cross_model_path):
        try:
            val = MosaicValidator(strips, cross_model_path, label="CrossTrack")
            val.run_validation(os.path.join(QC_DIR, "CrossTrack"))
        except Exception as e:
            print(f"Error CrossTrack: {e}")

    print("All Done.")