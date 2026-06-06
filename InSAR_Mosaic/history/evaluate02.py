# -*- coding: utf-8 -*-
"""
Script 2 (Final V21 Polyline-Cross): InSAR Mosaic Validator
功能：读取模型和掩膜，进行多维度的精确验证。
特性：
1. [双折线交互]：
   - 左键：加点（画折线）。
   - 右键：第一次按->结束Line1开始Line2；第二次按->全部结束。
2. [全指标统计]：RMSE, MAE, STD, MED。
3. [双图高清报表]：Visual + Stats (DPI=600)。
4. [闭环验证]：优先读取 Script 1 的 Mask。
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
import scipy.stats as stats
from scipy.stats import gaussian_kde

# =========================
# 0. 全局配置
# =========================
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'stix',
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlesize": 14,
    "figure.titlesize": 16
}
rcParams.update(config)

# --- 路径配置 ---
OUT_DIR = r"H:\\InSAR_Mosaic_Project"
MODEL_DIR = os.path.join(OUT_DIR, "Models")
RESULT_DIR = os.path.join(OUT_DIR, "Results")
MASK_DIR = os.path.join(OUT_DIR, "Masks")
QC_DIR = os.path.join(OUT_DIR, "Quality_Report_Final")

NODATA_VALUE = -9999.0
MAX_CORRECTION = 0.5

if not os.path.exists(QC_DIR): os.makedirs(QC_DIR)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# =========================
# 1. 辅助工具函数 (核心交互逻辑修改)
# =========================

class CrossPolylineDrawer:
    """
    [V21] 双折线交互工具
    逻辑：
    - 左键：加点
    - 右键：切换下一条线 或 结束
    """

    def __init__(self, ax):
        self.ax = ax
        self.line1_pts = []
        self.line2_pts = []
        self.active_line = 1  # 1=画第一条, 2=画第二条

        # 可视化对象
        self.vis_line1, = ax.plot([], [], 'r-o', linewidth=2, markersize=5, label='Line 1 (Active)')
        self.vis_line2, = ax.plot([], [], 'b-o', linewidth=2, markersize=5, label='Line 2', alpha=0.3)
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)

    def on_press(self, event):
        if event.inaxes != self.ax: return

        # --- 左键：加点 ---
        if event.button == 1:
            if self.active_line == 1:
                self.line1_pts.append((event.xdata, event.ydata))
            else:
                self.line2_pts.append((event.xdata, event.ydata))
            self.update_plot()

        # --- 右键：切换 或 结束 ---
        elif event.button == 3 or event.button == 2:
            if self.active_line == 1:
                if len(self.line1_pts) < 2:
                    print(">>> Line 1 点数太少，请继续加点。")
                else:
                    print(">>> Line 1 完成！现在开始画 Line 2 (蓝色)。")
                    self.active_line = 2
                    self.vis_line1.set_label('Line 1 (Done)')
                    self.vis_line2.set_label('Line 2 (Active)')
                    self.vis_line2.set_alpha(1.0)  # 激活显示
                    self.ax.legend()
                    self.ax.figure.canvas.draw()
            else:
                if len(self.line2_pts) < 2:
                    print(">>> Line 2 点数太少，请继续加点。")
                else:
                    print(">>> 全部绘制完成。")
                    plt.close(self.ax.figure)

    def update_plot(self):
        if self.line1_pts:
            xs, ys = zip(*self.line1_pts);
            self.vis_line1.set_data(xs, ys)
        if self.line2_pts:
            xs, ys = zip(*self.line2_pts);
            self.vis_line2.set_data(xs, ys)
        self.ax.figure.canvas.draw()


def interpolate_polyline(pts):
    """将多个折线点插值为密集坐标"""
    if len(pts) < 2: return np.array([]), np.array([])

    dense_r = []
    dense_c = []

    cols = [p[0] for p in pts]
    rows = [p[1] for p in pts]

    for k in range(len(pts) - 1):
        r0, c0 = rows[k], cols[k]
        r1, c1 = rows[k + 1], cols[k + 1]
        dist = np.hypot(r1 - r0, c1 - c0)
        num_steps = int(dist * 2.0)  # 2倍采样
        if num_steps < 2: num_steps = 2

        rr = np.linspace(r0, r1, num_steps)
        cc = np.linspace(c0, c1, num_steps)

        # 避免连接处重复点
        if k < len(pts) - 2:
            dense_r.extend(rr[:-1]);
            dense_c.extend(cc[:-1])
        else:
            dense_r.extend(rr);
            dense_c.extend(cc)

    return np.array(dense_r), np.array(dense_c)


def manual_draw_cross(diff_map, mask, title_info=""):
    """弹出窗口让用户画两条线"""
    fig_click, ax = plt.subplots(figsize=(12, 8))
    valid_vals = diff_map[mask]
    vmax = np.nanpercentile(np.abs(valid_vals), 98) if len(valid_vals) > 0 else 1.0
    display_img = np.where(mask, diff_map, np.nan)
    ax.imshow(display_img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    instr = (
        "INSTRUCTION:\n"
        "1. LEFT Click to add points for Line 1 (Red).\n"
        "2. RIGHT Click to FINISH Line 1 and START Line 2 (Blue).\n"
        "3. LEFT Click to add points for Line 2.\n"
        "4. RIGHT Click to FINISH ALL."
    )
    ax.set_title(f"{title_info}\n{instr}", color='red', fontweight='bold', fontsize=12)

    print(f"\n--- 交互模式: {title_info} ---")
    print(">>> 1. 左键加点画红线。")
    print(">>> 2. 画完红线按【右键】切换到蓝线。")
    print(">>> 3. 左键加点画蓝线。")
    print(">>> 4. 画完蓝线按【右键】结束。")

    drawer = CrossPolylineDrawer(ax)
    plt.show(block=True)

    r1, c1 = interpolate_polyline(drawer.line1_pts)
    r2, c2 = interpolate_polyline(drawer.line2_pts)
    return [(r1, c1), (r2, c2)]


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
    if len(x) > 15000:
        idx = np.random.choice(len(x), size=15000, replace=False)
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

    min_val = min(x_samp.min(), y_samp.min());
    max_val = max(x_samp.max(), y_samp.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.0, alpha=0.8)
    ax.scatter(x_samp, y_samp, c=z, s=5, cmap='Spectral_r', alpha=0.8, edgecolors='none')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)


# =========================
# 2. 验证器类
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
        self._init_geometry()

    def _init_geometry(self):
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
        csv_path = os.path.join(output_dir, f"{self.label}_Stats_Extended.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(
                    "Pair,Name1,Name2,Points_Valid,RMSE_Raw,RMSE_Corr,MAE_Raw,MAE_Corr,STD_Raw,STD_Corr,MED_Raw,MED_Corr,Imp_%,Status\n")

        logging.info(f"--- Starting Validation Group: {self.label} ---")
        for i in range(len(self.images)):
            for j in range(i + 1, len(self.images)):
                self._process_pair(i, j, output_dir, csv_path)

    def _process_pair(self, i, j, save_dir, csv_file):
        img1, img2 = self.images[i], self.images[j]
        name1 = os.path.basename(img1['vel']).rsplit('.', 1)[0]
        name2 = os.path.basename(img2['vel']).rsplit('.', 1)[0]

        with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2:
            xl = max(src1.bounds.left, src2.bounds.left);
            xr = min(src1.bounds.right, src2.bounds.right)
            yb = max(src1.bounds.bottom, src2.bounds.bottom);
            yt = min(src1.bounds.top, src2.bounds.top)
            if xr <= xl or yt <= yb: return
            width = int((xr - xl) / src1.res[0]);
            height = int((yt - yb) / src1.res[1])
            if width < 10 or height < 10: return
            trans = from_origin(xl, yt, src1.res[0], src1.res[1])
            prof = {'height': height, 'width': width, 'transform': trans, 'crs': src1.crs}
            v1 = reproject_to_grid(src1, prof);
            v2 = reproject_to_grid(src2, prof)

        rows, cols = np.indices(v1.shape)
        px, py = trans * (cols, rows)
        px -= self.ref_x;
        py -= self.ref_y

        def poly(p, x, y):
            return p[0] + p[1] * x + p[2] * y + p[3] * x ** 2 + p[4] * y ** 2 + p[5] * x * y

        c1 = poly(self.corrections[i], px, py);
        c2 = poly(self.corrections[j], px, py)
        v1_corr = v1 - c1;
        v2_corr = v2 - c2
        diff_raw = v1 - v2;
        diff_corr = v1_corr - v2_corr

        mask_basic = np.isfinite(diff_raw) & np.isfinite(diff_corr) & (v1 != NODATA_VALUE) & (v2 != NODATA_VALUE)
        if np.sum(mask_basic) < 300: return

        # --- 闭环掩膜加载 ---
        mask_file = os.path.join(MASK_DIR, f"Mask_{i}_{j}.npy")
        mask_final = None;
        status_tag = "Robust(Est)"
        if os.path.exists(mask_file):
            try:
                packed = np.load(mask_file)
                if packed.size * 8 >= v1.size:
                    loaded_mask = np.unpackbits(packed, count=v1.size).reshape(v1.shape).astype(bool)
                    mask_final = mask_basic & loaded_mask
                    if np.sum(mask_final) > 100:
                        status_tag = "Exact(Load)"
                    else:
                        mask_final = None
            except:
                pass

        if mask_final is None:
            resid = diff_corr[mask_basic]
            median_val = np.median(resid)
            mad_val = np.median(np.abs(resid - median_val))
            threshold = max(3 * 1.4826 * mad_val, 0.003)
            mask_inliers = mask_basic & (np.abs(diff_corr - median_val) < threshold)
            if np.sum(mask_inliers) < np.sum(mask_basic) * 0.2:
                mask_final = mask_basic; status_tag = "Raw(Poor)"
            else:
                mask_final = mask_inliers; status_tag = "Robust(Est)"

        # --- 统计指标 ---
        d_raw = diff_raw[mask_final]
        d_corr = diff_corr[mask_final]

        rmse_raw = np.sqrt(np.mean(d_raw ** 2));
        rmse_corr = np.sqrt(np.mean(d_corr ** 2))
        mae_raw = np.mean(np.abs(d_raw));
        mae_corr = np.mean(np.abs(d_corr))
        std_raw = np.std(d_raw);
        std_corr = np.std(d_corr)
        med_raw = np.median(np.abs(d_raw));
        med_corr = np.median(np.abs(d_corr))
        imp = (rmse_raw - rmse_corr) / (rmse_raw + 1e-9) * 100

        # 写入 CSV
        line = f"{i}-{j},{name1},{name2},{np.sum(mask_final)},{rmse_raw:.4f},{rmse_corr:.4f},{mae_raw:.4f},{mae_corr:.4f},{std_raw:.4f},{std_corr:.4f},{med_raw:.4f},{med_corr:.4f},{imp:.1f},{status_tag}\n"
        with open(csv_file, 'a') as f:
            f.write(line)

        # 交互画图
        title_tag = f"{name1}\nvs\n{name2}"
        lines_list = manual_draw_cross(diff_corr, mask_basic, title_info=title_tag)

        metrics = {
            'rmse_r': rmse_raw, 'rmse_c': rmse_corr,
            'mae_r': mae_raw, 'mae_c': mae_corr,
            'std_r': std_raw, 'std_c': std_corr,
            'med_r': med_raw, 'med_c': med_corr,
            'imp': imp, 'pts': np.sum(mask_final)
        }

        self._plot_dual_report(name1, name2, v1_corr, v2_corr, diff_corr, mask_basic,
                               v1[mask_final], v2[mask_final], v1_corr[mask_final], v2_corr[mask_final],
                               d_raw, d_corr, metrics, save_dir,
                               lines_list, status_tag, i, j)

    def _plot_dual_report(self, name1, name2, img1, img2, diff_map, mask_show,
                          raw_x, raw_y, corr_x, corr_y, d_raw, d_corr,
                          metrics, out_dir, lines_list, status_tag, idx1, idx2):

        # =================================================
        # FIGURE 1: Visual Inspection
        # =================================================
        fig1 = plt.figure(figsize=(20, 6))
        gs1 = fig1.add_gridspec(1, 3, wspace=0.1)

        combined_img_vals = np.concatenate([img1[mask_show], img2[mask_show]])
        try:
            vmin, vmax = np.percentile(combined_img_vals, [2, 98])
        except:
            vmin, vmax = -5, 5
        norm_common = Normalize(vmin=vmin, vmax=vmax)

        ax1 = fig1.add_subplot(gs1[0, 0])
        im1 = ax1.imshow(np.where(mask_show, img1, np.nan), cmap='jet', norm=norm_common)
        ax1.set_title(f"(a) Left: {name1}", fontweight='bold', fontsize=13);
        ax1.axis('off')

        ax2 = fig1.add_subplot(gs1[0, 1])
        im2 = ax2.imshow(np.where(mask_show, img2, np.nan), cmap='jet', norm=norm_common)
        ax2.set_title(f"(b) Right: {name2}", fontweight='bold', fontsize=13);
        ax2.axis('off')

        cax = fig1.add_axes([0.15, 0.05, 0.45, 0.03])
        plt.colorbar(im1, cax=cax, orientation='horizontal', label='Velocity (mm/yr)')

        d_max = max(np.percentile(np.abs(d_corr), 99), 0.5)
        norm_diff = Normalize(vmin=-d_max, vmax=d_max)
        ax3 = fig1.add_subplot(gs1[0, 2])
        im3 = ax3.imshow(np.where(mask_show, diff_map, np.nan), cmap='RdBu_r', norm=norm_diff)
        colors = ['r-', 'b-']
        for k, (line_r, line_c) in enumerate(lines_list):
            if len(line_r) > 1: ax3.plot(line_c, line_r, colors[k], linewidth=2.5, alpha=0.8)
        ax3.set_title(f"(c) Diff Map & Profiles", fontweight='bold', fontsize=13);
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, label='Diff (mm/yr)')

        plt.suptitle(f"Visual Check: {name1} vs {name2}", fontsize=18, fontweight='bold')
        fig1.savefig(os.path.join(out_dir, f"Check_{idx1}_{idx2}_1_Visual.png"), dpi=600, bbox_inches='tight')
        plt.close(fig1)

        # =================================================
        # FIGURE 2: Statistical Analysis
        # =================================================
        fig2 = plt.figure(figsize=(16, 12))
        gs2 = fig2.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        # (d)(e) Profiles
        short_n1 = (name1[:12] + '..') if len(name1) > 12 else name1
        short_n2 = (name2[:12] + '..') if len(name2) > 12 else name2

        for k in range(2):
            ax_prof = fig2.add_subplot(gs2[0, k])
            line_r, line_c = lines_list[k]
            if len(line_r) > 1:
                idx_r = np.clip(np.round(line_r).astype(int), 0, img1.shape[0] - 1)
                idx_c = np.clip(np.round(line_c).astype(int), 0, img1.shape[1] - 1)
                p_y1 = img1[idx_r, idx_c];
                p_y2 = img2[idx_r, idx_c]
                p_x = np.cumsum(np.insert(np.sqrt(np.diff(line_r) ** 2 + np.diff(line_c) ** 2), 0, 0))
                ax_prof.plot(p_x, p_y1, 'r-', linewidth=1.5, alpha=0.7, label=short_n1)
                ax_prof.plot(p_x, p_y2, 'b--', linewidth=1.5, alpha=0.9, label=short_n2)
                ax_prof.set_title(f"({chr(100 + k)}) Profile {k + 1}", fontweight='bold')
                if k == 1: ax_prof.legend(fontsize=9)
                ax_prof.grid(True, linestyle=':', alpha=0.6)
            else:
                ax_prof.text(0.5, 0.5, f"Profile {k + 1} Not Drawn", ha='center')

        # (f)(g) Scatters
        ax_sc1 = fig2.add_subplot(gs2[1, 0]);
        plot_density_scatter(ax_sc1, raw_x, raw_y, "(f) Scatter: Before")
        ax_sc2 = fig2.add_subplot(gs2[1, 1]);
        plot_density_scatter(ax_sc2, corr_x, corr_y, "(g) Scatter: After")

        # (h) Histogram
        ax_hist = fig2.add_subplot(gs2[2, 0])
        combined = np.concatenate([d_raw, d_corr])
        try:
            vmin, vmax = np.percentile(combined, [1, 99])
            bins = np.linspace(vmin, vmax, 60)
            ax_hist.hist(d_raw, bins=bins, density=True, color='gray', alpha=0.4, label='Raw')
            ax_hist.hist(d_corr, bins=bins, density=True, color='red', alpha=0.5, label='Corr')
        except:
            pass
        ax_hist.set_title(f"(h) Error Distribution", fontweight='bold');
        ax_hist.legend()

        # (i) Summary Table
        ax_sum = fig2.add_subplot(gs2[2, 1]);
        ax_sum.axis('off')

        def wrap(n):
            return n if len(n) < 30 else n[:15] + '...' + n[-10:]

        bg_color = "#e0ffe0" if "Exact" in status_tag else "#ffffe0"

        stats_text = (
            f"STATISTICAL SUMMARY\n"
            f"==========================================\n"
            f"Left Img : {wrap(name1)}\n"
            f"Right Img: {wrap(name2)}\n"
            f"Mode     : {status_tag}\n"
            f"Valid Pts: {metrics['pts']}\n"
            f"------------------------------------------\n"
            f"METRIC   |    RAW     |  CORRECTED \n"
            f"------------------------------------------\n"
            f"RMSE     |  {metrics['rmse_r']:.4f}    |  {metrics['rmse_c']:.4f}\n"
            f"MAE      |  {metrics['mae_r']:.4f}    |  {metrics['mae_c']:.4f}\n"
            f"Std Dev  |  {metrics['std_r']:.4f}    |  {metrics['std_c']:.4f}\n"
            f"Median E |  {metrics['med_r']:.4f}    |  {metrics['med_c']:.4f}\n"
            f"------------------------------------------\n"
            f"IMPROVEMENT: {metrics['imp']:.1f} %\n"
            f"=========================================="
        )

        ax_sum.text(0.05, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=1", facecolor=bg_color, edgecolor="black"))

        plt.suptitle(f"Statistical Report: {name1} vs {name2}", fontsize=18, fontweight='bold')
        fig2.savefig(os.path.join(out_dir, f"Check_{idx1}_{idx2}_2_Stats.png"), dpi=600, bbox_inches='tight')
        plt.close(fig2)


# =========================
# 入口 (Main)
# =========================
if __name__ == "__main__":
    # --- 输入数据 ---
    inputs = [
        {'id': 0, 'vel': r"E:\shanxi\18\18_113_126\vel_18_113_126.tif", 'coh': r"E:\shanxi\18\18_113_126\coh_18_113_126.tif"},
        {'id': 1, 'vel': r"E:\shanxi\18\18_113_121\vel_18_113_121.tif", 'coh': r"E:\shanxi\18\18_113_121\coh_18_113_121.tif"},
        {'id': 2, 'vel': r"E:\shanxi\18\18_113_116\vel_18_113_116.tif", 'coh': r"E:\shanxi\18\18_113_116\coh_18_113_116.tif"},
        {'id': 3, 'vel': r"E:\shanxi\18\18_113_111\vel_18_113_111.tif", 'coh': r"E:\shanxi\18\18_113_111\coh_18_113_111.tif"},
        {'id': 4, 'vel': r"E:\shanxi\18\18_11_121\vel_18_11_121.tif", 'coh': r"E:\shanxi\18\18_11_121\coh_18_11_121.tif"},
        {'id': 5, 'vel': r"E:\shanxi\18\18_11_116\vel_18_11_116.tif", 'coh': r"E:\shanxi\18\18_11_116\coh_18_11_116.tif"},
        {'id': 6, 'vel': r"E:\shanxi\18\18_11_111\vel_18_11_111.tif", 'coh': r"E:\shanxi\18\18_11_111\coh_18_11_111.tif"},
        {'id': 7, 'vel': r"E:\shanxi\18\18_40_127\vel_18_40_127.tif", 'coh': r"E:\shanxi\18\18_40_127\coh_18_40_127.tif"},
        {'id': 8, 'vel': r"E:\shanxi\18\18_40_122\vel_18_40_122.tif", 'coh': r"E:\shanxi\18\18_40_122\coh_18_40_122.tif"},
        {'id': 9, 'vel': r"E:\shanxi\18\18_40_117\vel_18_40_117.tif", 'coh': r"E:\shanxi\18\18_40_117\coh_18_40_117.tif"},
    ]

    tracks = {}
    pattern = re.compile(r'vel_\d+_(\d+)_\d+')
    for item in inputs:
        filename = os.path.basename(item['vel'])
        match = pattern.search(filename)
        path_id = match.group(1) if match else "Unknown"
        if path_id not in tracks: tracks[path_id] = []
        tracks[path_id].append(item)

    print("\n" + "=" * 50);
    print("STEP 1: 同轨验证");
    print("=" * 50)
    for pid, imgs in tracks.items():
        if len(imgs) < 2: continue
        model_path = os.path.join(MODEL_DIR, f"Track_{pid}_model.npy")
        if os.path.exists(model_path):
            try:
                val = MosaicValidator(imgs, model_path, label=f"Track_{pid}")
                val.run_validation(os.path.join(QC_DIR, f"Track_{pid}"))
            except Exception as e:
                print(f"Err: {e}")

    print("\n" + "=" * 50);
    print("STEP 2: 异轨验证");
    print("=" * 50)
    strip_vels = glob.glob(os.path.join(RESULT_DIR, "Strip_*_vel.tif"))
    strips = []
    for v_path in strip_vels:
        c_path = v_path.replace("_vel.tif", "_coh.tif")
        if os.path.exists(c_path): strips.append({'vel': v_path, 'coh': c_path})

    cross_model_path = os.path.join(MODEL_DIR, "CrossTrack_model.npy")
    if len(strips) >= 2 and os.path.exists(cross_model_path):
        try:
            val = MosaicValidator(strips, cross_model_path, label="CrossTrack")
            val.run_validation(os.path.join(QC_DIR, "CrossTrack"))
        except Exception as e:
            print(f"Err: {e}")

    print(f"\nAll Done. Reports saved to: {QC_DIR}")