import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ==========================================
# 全局字体设置：使用 Times New Roman
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示异常

# ==========================================
# 1. 填入您的文件夹路径和图片保存路径
# ==========================================
FOLDER_PATH = r"D:\BaiduSyncdisk\课题首要\Mosaic_InSAR\photos\analysis\incidenceAngle"
SAVE_PATH = r"D:\BaiduSyncdisk\课题首要\Mosaic_InSAR\photos\analysis\incidenceAngle\InSAR_Result.png"

# 自动获取文件夹下的前 4 个 .tif 文件
tif_files = glob.glob(os.path.join(FOLDER_PATH, "*.tif"))[:4]

if not tif_files:
    print(f"在 {FOLDER_PATH} 下没有找到 .tif 文件！请检查路径。")
else:
    # 创建 2x2 的子图网格
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
    axes = axes.flatten()

    # ==========================================
    # 2. 强制指定范围：30 到 50
    # ==========================================
    VMIN = 30.0
    VMAX = 50.0

    for i in range(4):
        ax = axes[i]

        # 隐藏多余的子图
        if i >= len(tif_files):
            ax.axis('off')
            continue

        filepath = tif_files[i]
        filename = os.path.basename(filepath)

        # 读取 TIF 数据
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(float)

            # 处理 NoData 和常见的 -9999.0 背景值
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            data[data == -9999.0] = np.nan

            # 3. 绘制影像，强制使用 VMIN=30 和 VMAX=50
        im = ax.imshow(data, cmap='RdBu_r', vmin=VMIN, vmax=VMAX)

        # 添加标题
        ax.set_title(f"({chr(97 + i)}) {filename}", fontweight='bold', fontsize=14)
        ax.axis('off')

        # 4. 画出颜色条 (Colorbar)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity (mm/yr)', fontsize=12)
        cbar.ax.tick_params(labelsize=11)  # 调整颜色条上的数字大小

    # 调整整体布局
    plt.tight_layout()
    plt.suptitle(f"InSAR Velocity (Fixed Scale: {VMIN} to {VMAX} mm/yr)", fontsize=18, fontweight='bold', y=1.02)

    # ==========================================
    # 5. 保存图片到本地 (300 DPI)
    # ==========================================
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f">>> 图片已成功保存至: {SAVE_PATH} (300 DPI)")

    # 在屏幕上显示图片
    plt.show()