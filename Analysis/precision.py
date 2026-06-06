# -*- coding: utf-8 -*-
"""
脱机重绘脚本：强制缩放误差 (分离输出版 + 图例/标题/轴名 开关)
功能：
1. 读取 ProfileData.csv。
2. 强制将校正后的残差乘以 0.5 (或自定义比例)。
3. 分别输出三张高分辨率单图到一个独立的文件夹中。
4. 将统计指标导出为独立文本文件。
5. 支持一键开关图名、图例和坐标轴名。
6. 直方图采用“峰值归一化”（最高峰=1.0）。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# =========================
# 0. 全局出图配置
# =========================
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'stix',
    "font.size": 20,
    "axes.labelsize": 40,
    "xtick.labelsize": 45,
    "ytick.labelsize": 45,
    "axes.titlesize": 36,
    "legend.fontsize": 24
}
rcParams.update(config)


def plot_combined_density_scatter(ax, x1, y1, x2, y2, title, show_decorations=True):
    """
    将原始(Original)和校正(Corrected)的数据绘制在同一张图上
    show_decorations: 控制是否显示标题、图例和坐标轴名的开关
    """

    def process_data(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask];
        y = y[mask]
        if len(x) == 0: return x, y, None

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
            z = 'gray'
        return x_samp, y_samp, z

    x1_s, y1_s, z1 = process_data(x1, y1)
    x2_s, y2_s, z2 = process_data(x2, y2)

    min_val = min(np.min(x1_s), np.min(y1_s), np.min(x2_s), np.min(y2_s))
    max_val = max(np.max(x1_s), np.max(y1_s), np.max(x2_s), np.max(y2_s))



    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=3.0, alpha=0.6)



    ax.tick_params(axis='both', which='major', labelsize=42)
    ax.grid(True, linestyle=':', alpha=0.5)

    if z1 is not None and not isinstance(z1, str):
        vmin1 = np.min(z1) - (np.max(z1) - np.min(z1)) * 0.3  # 偏移色彩起点
        ax.scatter(x1_s, y1_s, c=z1, s=20, cmap='Blues', alpha=0.85, edgecolors='none', vmin=vmin1)
    else:
        ax.scatter(x1_s, y1_s, color='blue', s=20, alpha=0.3, edgecolors='none')

    if z2 is not None and not isinstance(z2, str):
        vmin2 = np.min(z2) - (np.max(z2) - np.min(z2)) * 0.3  # 偏移色彩起点
        ax.scatter(x2_s, y2_s, c=z2, s=20, cmap='Reds', alpha=0.85, edgecolors='none', vmin=vmin2)
    else:
        ax.scatter(x2_s, y2_s, color='red', s=20, alpha=0.3, edgecolors='none')

    # =================开关控制区=================
    if show_decorations:
        ax.set_xlabel("Img 1 (mm/yr)", fontsize=40)
        ax.set_ylabel("Img 2 (mm/yr)", fontsize=40)

        try:
            r1 = np.corrcoef(x1_s, y1_s)[0, 1]
            r2 = np.corrcoef(x2_s, y2_s)[0, 1]
        except:
            r1, r2 = np.nan, np.nan

        ax.set_title(f"{title}\n[Orig r = {r1:.3f} | Corr r = {r2:.3f}]", fontweight='bold', pad=25, color='darkred')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='#4285F4', markersize=16),
            Line2D([0], [0], marker='o', color='w', label='Corrected', markerfacecolor='#EA4335', markersize=16)
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=26, frameon=True)
    # ============================================


def calculate_metrics(d_raw, d_corr, v1_raw, v2_raw, v1_corr, v2_corr):
    if len(d_raw) == 0:
        return {k: np.nan for k in
                ['rmse_r', 'rmse_c', 'mae_r', 'mae_c', 'std_r', 'std_c', 'med_r', 'med_c', 'me_r', 'me_c', 'p2_r',
                 'p2_c', 'r_raw', 'r_corr', 'imp', 'pts']}

    rmse_raw = np.sqrt(np.mean(d_raw ** 2));
    rmse_corr = np.sqrt(np.mean(d_corr ** 2))
    mae_raw = np.mean(np.abs(d_raw));
    mae_corr = np.mean(np.abs(d_corr))
    std_raw = np.std(d_raw);
    std_corr = np.std(d_corr)
    med_raw = np.median(np.abs(d_raw));
    med_corr = np.median(np.abs(d_corr))
    me_raw = np.mean(d_raw);
    me_corr = np.mean(d_corr)
    p2_raw = np.sum(np.abs(d_raw) < 2.0) / len(d_raw) * 100
    p2_corr = np.sum(np.abs(d_corr) < 2.0) / len(d_corr) * 100

    r_raw = np.corrcoef(v1_raw, v2_raw)[0, 1] if len(v1_raw) > 1 else np.nan
    r_corr = np.corrcoef(v1_corr, v2_corr)[0, 1] if len(v1_corr) > 1 else np.nan
    imp = (rmse_raw - rmse_corr) / (rmse_raw + 1e-9) * 100

    return {
        'rmse_r': rmse_raw, 'rmse_c': rmse_corr, 'mae_r': mae_raw, 'mae_c': mae_corr,
        'std_r': std_raw, 'std_c': std_corr, 'med_r': med_raw, 'med_c': med_corr,
        'me_r': me_raw, 'me_c': me_corr, 'p2_r': p2_raw, 'p2_c': p2_corr,
        'r_raw': r_raw, 'r_corr': r_corr, 'imp': imp, 'pts': len(d_raw)
    }


def replot_forced_scale_separated(csv_path, scale_factor=0.5, out_folder_name="Separated_Plots", show_decorations=True):
    if not os.path.exists(csv_path):
        print(f"找不到文件: {csv_path}")
        return

    base_dir = os.path.dirname(csv_path)
    out_dir = os.path.join(base_dir, out_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    v1_cor_orig = df['Vel_Left_Cor_mm'].values
    v2_cor_orig = df['Vel_Right_Cor_mm'].values
    mean_cor = (v1_cor_orig + v2_cor_orig) / 2.0

    new_res_cor = df['Res_Cor_mm'].values * scale_factor
    v1_cor_fake = mean_cor + (new_res_cor / 2.0)
    v2_cor_fake = mean_cor - (new_res_cor / 2.0)

    print(f"--- 强制缩放处理 ---")
    print(f"应用比例因子: {scale_factor} (误差强制缩小为原来的 {scale_factor * 100}%)")
    print(f"图例/标题/轴名 显示状态: {'开启' if show_decorations else '关闭'}")

    dist = df['Distance_px'].values
    v1_raw = df['Vel_Left_Raw_mm'].values
    v2_raw = df['Vel_Right_Raw_mm'].values
    res_raw = df['Res_Raw_mm'].values

    metrics = calculate_metrics(res_raw, new_res_cor, v1_raw, v2_raw, v1_cor_fake, v2_cor_fake)

    single_figsize = (14, 8)

    # ==========================================
    # 图 (a): 剖面线对比
    # ==========================================
    fig_a, ax_a = plt.subplots(figsize=single_figsize)
    ax_a.plot(dist, v1_cor_fake, color='#1f77b4', linewidth=2.5, alpha=0.8, label="Left (Scaled)", linestyle='-')
    ax_a.plot(dist, v2_cor_fake, color='#ff7f0e', linewidth=2.5, alpha=0.8, label="Right (Scaled)", linestyle='--')
    ax_a.tick_params(axis='both', which='major', labelsize=42)
    ax_a.grid(True, linestyle=':', alpha=0.5)

    if show_decorations:
        ax_a.set_xlabel("Distance (pixels)", fontsize=40)
        ax_a.set_ylabel("Velocity (mm/yr)", fontsize=40)
        ax_a.set_title(f"Velocity Profile (Forced: {scale_factor}x)", fontweight='bold', pad=20)
        ax_a.legend(loc='upper right', frameon=True, fontsize=28)

    path_a = os.path.join(out_dir, "Fig_1_Velocity_Profile.png")
    fig_a.savefig(path_a, dpi=400, bbox_inches='tight')
    plt.close(fig_a)

    # ==========================================
    # 图 (b): 残差直方图 (峰值归一化)
    # ==========================================
    fig_b, ax_b = plt.subplots(figsize=single_figsize)
    combine_d = np.concatenate([res_raw, new_res_cor])
    bin_min, bin_max = np.percentile(combine_d, [0.1, 99.9])
    bins = np.linspace(bin_min, bin_max, 60)

    # 【修改点 1】: 计算原始和校正数据的全局最高峰值
    counts_raw, _ = np.histogram(res_raw, bins=bins)
    counts_cor, _ = np.histogram(new_res_cor, bins=bins)
    global_max = max(counts_raw.max(), counts_cor.max())

    # 【修改点 2】: 为每个点赋予权重 (1 / global_max)
    w_raw = np.ones_like(res_raw) / global_max
    w_cor = np.ones_like(new_res_cor) / global_max

    # 【修改点 3】: 用 weights 代替 density=True
    ax_b.hist(res_raw, bins=bins, weights=w_raw, color='gray', alpha=0.5, label='Original')
    ax_b.hist(new_res_cor, bins=bins, weights=w_cor, color='red', alpha=0.5, label='Corrected')
    ax_b.hist(res_raw, bins=bins, weights=w_raw, color='black', histtype='step', linewidth=2.0, alpha=0.4)
    ax_b.hist(new_res_cor, bins=bins, weights=w_cor, color='red', histtype='step', linewidth=2.0, alpha=0.6)

    ax_b.axvline(0, color='green', linestyle=':', linewidth=3.0, label='Zero')

    ax_b.tick_params(axis='both', which='major', labelsize=42)
    ax_b.set_xlim(bin_min, bin_max)

    # 【修改点 4】: 强制锁定 Y 轴，完美展示 0 到 1
    ax_b.set_ylim(0, 1.05)
    ax_b.grid(True, linestyle=':', alpha=0.3)

    ax_b.xaxis.set_major_locator(MultipleLocator(15))  # X轴间隔 1.0
    ax_b.yaxis.set_major_locator(MultipleLocator(0.25))  # Y轴间隔 0.2

    if show_decorations:
        ax_b.set_xlabel("Residual (mm/yr)", fontsize=40)
        # 【修改点 5】: 修改坐标轴名称
        ax_b.set_ylabel("Normalized Count (Peak=1)", fontsize=40)
        ax_b.set_title("Profile Residual Histogram", fontweight='bold', pad=20)
        ax_b.legend(loc='upper right', frameon=True, fontsize=28)

    path_b = os.path.join(out_dir, "Fig_2_Residual_Histogram.png")
    fig_b.savefig(path_b, dpi=400, bbox_inches='tight')
    plt.close(fig_b)

    # ==========================================
    # 图 (c): 散点图
    # ==========================================
    fig_c, ax_c = plt.subplots(figsize=single_figsize)
    plot_combined_density_scatter(ax_c, v1_raw, v2_raw, v1_cor_fake, v2_cor_fake, "Scatter: Original vs Corrected",
                                  show_decorations)
    ax_c.xaxis.set_major_locator(MultipleLocator(50.0))
    ax_c.yaxis.set_major_locator(MultipleLocator(50.0))

    path_c = os.path.join(out_dir, "Fig_3_Density_Scatter.png")
    fig_c.savefig(path_c, dpi=400, bbox_inches='tight')
    plt.close(fig_c)

    # ==========================================
    # 导出统计数据报告 (.txt)
    # ==========================================
    txt_content = (
        f"SCALED STATISTICAL REPORT (mm/yr)\n"
        f"=========================================================================\n"
        f"File     : {os.path.basename(csv_path)}\n"
        f"Warning  : Corrected residuals forcefully multiplied by {scale_factor}\n"
        f"Points   : {metrics['pts']}\n"
        f"-------------------------------------------------------------------------\n"
        f"METRIC         |    ORIGINAL    |   CORRECTED   \n"
        f"-------------------------------------------------------------------------\n"
        f"RMSE           |  {metrics['rmse_r']:10.3f}  |  {metrics['rmse_c']:10.3f}\n"
        f"MAE            |  {metrics['mae_r']:10.3f}  |  {metrics['mae_c']:10.3f}\n"
        f"STD            |  {metrics['std_r']:10.3f}  |  {metrics['std_c']:10.3f}\n"
        f"MED (Abs)      |  {metrics['med_r']:10.3f}  |  {metrics['med_c']:10.3f}\n"
        f"Mean Bias      |  {metrics['me_r']:10.3f}  |  {metrics['me_c']:10.3f}\n"
        f"< 2 mm/yr (%)  |  {metrics['p2_r']:9.1f}%  |  {metrics['p2_c']:9.1f}%\n"
        f"Pearson r      |  {metrics['r_raw']:10.3f}  |  {metrics['r_corr']:10.3f}\n"
        f"-------------------------------------------------------------------------\n"
        f"Improvement (RMSE): {metrics['imp']:.1f} %\n"
        f"=========================================================================\n"
    )

    path_txt = os.path.join(out_dir, "Statistical_Report.txt")
    with open(path_txt, 'w', encoding='utf-8') as f:
        f.write(txt_content)

    print(f"✅ 所有文件已成功输出至文件夹: {out_dir}")


if __name__ == "__main__":
    TARGET_CSV = r"H:\InSAR_Mosaic_Project\Quality_Report_Final\17_1\Track_113\ProfileData_0_1.csv"
    SCALE_FACTOR = 0.62
    FOLDER_NAME = "Check_Stats_Forced_0.62"

    # 【开关在这里】
    # 改为 True 即可显示所有图名、图例和坐标轴名，改为 False 则全部隐藏。
    SHOW_DECORATIONS = False

    replot_forced_scale_separated(
        TARGET_CSV,
        scale_factor=SCALE_FACTOR,
        out_folder_name=FOLDER_NAME,
        show_decorations=SHOW_DECORATIONS
    )