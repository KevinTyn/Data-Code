# -*- coding: utf-8 -*-
"""
交互式 TIF 全局汇总对比工具 (SCI 出版级图表定制版 - 图例外置)
特点：
1. 图例强制外置 (坐标轴上方，水平排列)，绝不遮挡数据。
2. 学术配色：深海蓝 (Method 1) vs 砖红 (Method 2)
3. 深度信息：箱体内嵌亮黄色均值菱形 (Mean)，白色粗线为中位数 (Median)
"""

import os
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
from matplotlib import rcParams

# =========================
# 全局图表样式配置 (学术期刊标准)
# =========================
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'stix',
    "font.size": 18,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.titlesize": 28,
    "legend.fontsize": 20,
    "axes.linewidth": 2.0,
    "lines.linewidth": 2.0
}
rcParams.update(config)
NODATA_VALUE = -9999.0

# 经典学术期刊配色
COLOR_A = '#2878b5'  # 深海蓝 (Navy Blue)
COLOR_B = '#c82423'  # 砖红 (Brick Red)


class GlobalAcademicAnalyzer:
    def __init__(self, m1_paths, m2_paths, unit, out_dir="Academic_Summary_Results"):
        self.paths = m1_paths + m2_paths
        self.unit = unit
        self.out_dir = out_dir

        print("正在读取 4 张影像并进行重采样对齐，请稍候...")
        (self.arr_a1, self.arr_a2, self.arr_b1, self.arr_b2,
         self.valid_mask, self.transform, self.extent) = self._prepare_all_data(self.paths)

        if self.arr_a1 is None: return

        # [自动单位转换] 转换为目标单位 (乘以 1000 变 mm)
        self.arr_a1[self.valid_mask] *= 1000.0
        self.arr_a2[self.valid_mask] *= 1000.0
        self.arr_b1[self.valid_mask] *= 1000.0
        self.arr_b2[self.valid_mask] *= 1000.0

        self.diff_m1_arr = self.arr_a1 - self.arr_a2
        self.diff_m2_arr = self.arr_b1 - self.arr_b2

        self.rois_coordinates = []
        self.drawn_rects = []

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('SCI 级全局汇总分析工具')

        # 计算第一张图在重叠区的显示极值（使用 98% 分位数去除极端异常值）
        valid_data_a1 = self.arr_a1[self.valid_mask]
        v_max = np.percentile(np.abs(valid_data_a1), 98)
        if np.isnan(v_max) or v_max == 0: v_max = 10.0

        # 核心修改：将 self.diff_m1_arr 改为 self.arr_a1，仅显示有效重叠区
        im = self.ax.imshow(np.where(self.valid_mask, self.arr_a1, np.nan),
                            extent=self.extent, cmap='RdBu_r', vmin=-v_max, vmax=v_max)

        self.ax.set_title("请画框标记 ROI。\n按 [a] 确认添加；按 [Enter] 结束并生成学术汇总图。",
                          fontweight='bold', pad=15, fontsize=14)

        # 将 colorbar 的标签改为当前显示的图
        plt.colorbar(im, ax=self.ax, label=f'Image 1 in Overlap ({unit})')

        self.rs = RectangleSelector(self.ax, self._on_select, useblit=True, button=[1], minspanx=5, minspany=5,
                                    spancoords='pixels', interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        plt.show()

    def _prepare_all_data(self, paths):
        with rasterio.open(paths[0]) as src_ref:
            bounds_list = [rasterio.open(p).bounds for p in paths]
            xl, xr = max(b.left for b in bounds_list), min(b.right for b in bounds_list)
            yb, yt = max(b.bottom for b in bounds_list), min(b.top for b in bounds_list)
            if xr <= xl or yt <= yb: return [None] * 7
            res_x, res_y = src_ref.res
            width, height = int((xr - xl) / res_x), int((yt - yb) / res_y)
            trans = from_origin(xl, yt, res_x, res_y)
            prof = {'height': height, 'width': width, 'transform': trans, 'crs': src_ref.crs}

            def resample_one(path, profile):
                with rasterio.open(path) as src:
                    arr = np.empty((profile['height'], profile['width']), dtype=np.float32)
                    reproject(rasterio.band(src, 1), arr, src_transform=src.transform, src_crs=src.crs,
                              dst_transform=profile['transform'], dst_crs=profile['crs'],
                              resampling=Resampling.bilinear, dst_nodata=np.nan)
                    return arr

            arr_a1, arr_a2 = resample_one(paths[0], prof), resample_one(paths[1], prof)
            arr_b1, arr_b2 = resample_one(paths[2], prof), resample_one(paths[3], prof)

        valid = (np.isfinite(arr_a1) & np.isfinite(arr_a2) & np.isfinite(arr_b1) & np.isfinite(arr_b2) &
                 (arr_a1 != NODATA_VALUE) & (arr_a2 != NODATA_VALUE) & (arr_b1 != NODATA_VALUE) & (
                             arr_b2 != NODATA_VALUE))
        return arr_a1, arr_a2, arr_b1, arr_b2, valid, trans, [xl, xr, yb, yt]

    def _on_select(self, eclick, erelease):
        xmin, xmax = min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata)
        ymin, ymax = min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata)
        self.current_selection = (xmin, xmax, ymin, ymax)

    def _on_key_press(self, event):
        if event.key == 'a':
            if hasattr(self, 'current_selection'):
                self.rois_coordinates.append(self.current_selection)
                xmin, xmax, ymin, ymax = self.current_selection
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red',
                                         facecolor='none')
                self.ax.add_patch(rect)
                self.drawn_rects.append(rect)
                self.fig.canvas.draw()
                print(f"-> 添加第 {len(self.rois_coordinates)} 个选区")
                del self.current_selection
        elif event.key == 'enter':
            if not self.rois_coordinates: return
            plt.close(self.fig)
            self._compute_global_summary()

    def _compute_global_summary(self):
        print(f"\n[*] 正在后台计算 {len(self.rois_coordinates)} 个 ROI 的汇总数据...")
        os.makedirs(self.out_dir, exist_ok=True)

        roi_labels = []
        rmse_m1_list, rmse_m2_list = [], []
        mae_m1_list, mae_m2_list = [], []
        diff_m1_all_rois, diff_m2_all_rois = [], []
        points_count = []

        for i, (xmin, xmax, ymin, ymax) in enumerate(self.rois_coordinates, 1):
            col_min, row_max = ~self.transform * (xmin, ymin)
            col_max, row_min = ~self.transform * (xmax, ymax)
            r1, r2 = max(0, int(row_min)), min(self.diff_m1_arr.shape[0], int(row_max))
            c1, c2 = max(0, int(col_min)), min(self.diff_m1_arr.shape[1], int(col_max))

            sub_mask = self.valid_mask[r1:r2, c1:c2]
            diff_m1 = self.diff_m1_arr[r1:r2, c1:c2][sub_mask]
            diff_m2 = self.diff_m2_arr[r1:r2, c1:c2][sub_mask]

            n_pts = len(diff_m1)
            if n_pts < 10: continue

            roi_labels.append(f"ROI {i}")
            points_count.append(n_pts)

            rmse_m1_list.append(np.sqrt(np.mean(diff_m1 ** 2)))
            rmse_m2_list.append(np.sqrt(np.mean(diff_m2 ** 2)))
            mae_m1_list.append(np.mean(np.abs(diff_m1)))
            mae_m2_list.append(np.mean(np.abs(diff_m2)))

            if n_pts > 10000:
                idx = np.random.choice(n_pts, 10000, replace=False)
                diff_m1_all_rois.append(diff_m1[idx])
                diff_m2_all_rois.append(diff_m2[idx])
            else:
                diff_m1_all_rois.append(diff_m1)
                diff_m2_all_rois.append(diff_m2)

        if not roi_labels:
            print("没有足够的有效像元，程序结束。")
            return

        x = np.arange(len(roi_labels))
        width = 0.35

        # ================= 图 1: RMSE 精度对比分组柱状图 =================
        fig_bar, ax_bar = plt.subplots(figsize=(max(12, len(roi_labels) * 1.2), 7))
        rects1 = ax_bar.bar(x - width / 2, rmse_m1_list, width, label='Method 1', color=COLOR_A, alpha=0.85,
                            edgecolor='black', linewidth=1.5)
        rects2 = ax_bar.bar(x + width / 2, rmse_m2_list, width, label='Method 2', color=COLOR_B, alpha=0.85,
                            edgecolor='black', linewidth=1.5)

        ax_bar.set_ylabel(f'RMSE ({self.unit})', fontweight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(roi_labels, rotation=45 if len(roi_labels) > 5 else 0)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.6)

        # 【修改点1】：将图例放在坐标轴外面的正上方，并排显示
        ax_bar.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        # 加大 title 的 pad，留出空间给外置图例
        ax_bar.set_title('Root Mean Square Error Comparison Across ROIs', fontweight='bold', pad=45)

        for spine in ax_bar.spines.values(): spine.set_linewidth(2.0)

        # 使用 bbox_inches='tight' 确保外置的图例和标题不被裁剪
        fig_bar.savefig(os.path.join(self.out_dir, "Academic_1_RMSE_BarChart.png"), dpi=400, bbox_inches='tight')
        plt.close(fig_bar)

        # ================= 图 2: 残差分布分组箱线图 =================
        fig_box, ax_box = plt.subplots(figsize=(max(14, len(roi_labels) * 1.5), 8))

        boxprops_A = dict(facecolor=COLOR_A, alpha=0.8, edgecolor='black', linewidth=1.5)
        boxprops_B = dict(facecolor=COLOR_B, alpha=0.8, edgecolor='black', linewidth=1.5)
        medianprops = dict(color='white', linewidth=2.5)
        meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='#FFD700', markersize=8)
        whiskerprops = dict(color='black', linewidth=1.5, linestyle='--')
        capprops = dict(color='black', linewidth=2.0)

        bplot1 = ax_box.boxplot(diff_m1_all_rois, positions=x - 0.2, widths=0.3,
                                patch_artist=True, boxprops=boxprops_A,
                                medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,
                                showmeans=True, meanprops=meanprops, showfliers=False)

        bplot2 = ax_box.boxplot(diff_m2_all_rois, positions=x + 0.2, widths=0.3,
                                patch_artist=True, boxprops=boxprops_B,
                                medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops,
                                showmeans=True, meanprops=meanprops, showfliers=False)

        tolerance_val = 2.0
        ax_box.axhspan(-tolerance_val, tolerance_val, color='#2ca02c', alpha=0.1, zorder=0)
        ax_box.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, zorder=1)

        all_data = np.concatenate(diff_m1_all_rois + diff_m2_all_rois)
        y_min_limit, y_max_limit = np.percentile(all_data, [2, 98])
        y_padding = (y_max_limit - y_min_limit) * 0.15
        ax_box.set_ylim(y_min_limit - y_padding, y_max_limit + y_padding)

        ax_box.set_xticks(x)
        ax_box.set_xticklabels(roi_labels, rotation=45 if len(roi_labels) > 5 else 0)
        ax_box.set_ylabel(f'Residuals ({self.unit})', fontweight='bold')
        ax_box.grid(axis='y', linestyle='--', alpha=0.5)

        import matplotlib.lines as mlines
        mean_marker = mlines.Line2D([], [], color='w', marker='D', markerfacecolor='#FFD700', markeredgecolor='black',
                                    markersize=10)

        # 【修改点2】：将箱线图的图例放在坐标轴外面的正上方，3个元素一字排开
        ax_box.legend([bplot1["boxes"][0], bplot2["boxes"][0], mean_marker],
                      ['Method 1', 'Method 2', 'Mean (Bias)'],
                      loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        # 同样加大 title 的间距
        ax_box.set_title('Residual Boxplot (Tolerance Zone Highlighted)', fontweight='bold', pad=45)

        for spine in ax_box.spines.values(): spine.set_linewidth(2.0)

        # 确保 bbox_inches='tight' 以完整保存外部图例
        fig_box.savefig(os.path.join(self.out_dir, "Academic_2_Residual_Boxplot.png"), dpi=400, bbox_inches='tight')
        plt.close(fig_box)

        # ================= 导出 CSV 汇总长表 =================
        csv_path = os.path.join(self.out_dir, "Academic_Statistical_Summary.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("ROI_Name,Points_Count,RMSE_M1,RMSE_M2,MAE_M1,MAE_M2\n")
            for i in range(len(roi_labels)):
                f.write(
                    f"{roi_labels[i]},{points_count[i]},{rmse_m1_list[i]:.4f},{rmse_m2_list[i]:.4f},{mae_m1_list[i]:.4f},{mae_m2_list[i]:.4f}\n")

        print(f"✅ 学术级图表汇总完毕！请前往文件夹查看：{os.path.abspath(self.out_dir)}")

        # ================= 导出 CSV 汇总长表 =================
        csv_path = os.path.join(self.out_dir, "Academic_Statistical_Summary.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("ROI_Name,Points_Count,RMSE_M1,RMSE_M2,MAE_M1,MAE_M2\n")
            for i in range(len(roi_labels)):
                f.write(
                    f"{roi_labels[i]},{points_count[i]},{rmse_m1_list[i]:.4f},{rmse_m2_list[i]:.4f},{mae_m1_list[i]:.4f},{mae_m2_list[i]:.4f}\n")

        # ================= 【新增】导出 ROI 空间坐标范围 =================
        roi_coords_path = os.path.join(self.out_dir, "Academic_ROI_Coordinates.csv")
        with open(roi_coords_path, 'w', encoding='utf-8') as f:
            f.write("ROI_Name,X_Min,X_Max,Y_Min,Y_Max\n")
            for i, (xmin, xmax, ymin, ymax) in enumerate(self.rois_coordinates, 1):
                f.write(f"ROI {i},{xmin:.6f},{xmax:.6f},{ymin:.6f},{ymax:.6f}\n")

        print(f"✅ 学术级图表汇总完毕！请前往文件夹查看：{os.path.abspath(self.out_dir)}")

        # ================= 【新增】导出 ROI 为 Shapefile =================
        try:
            import geopandas as gpd
            from shapely.geometry import box
            import rasterio

            # 直接从第一张参考影像读取坐标系 (CRS)
            with rasterio.open(self.paths[0]) as src:
                roi_crs = src.crs

            geometries = []
            roi_names_shp = []
            for i, (xmin, xmax, ymin, ymax) in enumerate(self.rois_coordinates, 1):
                # 依据对角坐标生成矩形几何体
                geom = box(xmin, ymin, xmax, ymax)
                geometries.append(geom)
                roi_names_shp.append(f"ROI_{i}")

            if geometries:
                # 组装为地理数据框 (GeoDataFrame)
                gdf = gpd.GeoDataFrame({'ROI_Name': roi_names_shp}, geometry=geometries, crs=roi_crs)

                # 保存为 ESRI Shapefile
                shp_path = os.path.join(self.out_dir, "Academic_ROIs.shp")
                gdf.to_file(shp_path, driver="ESRI Shapefile", encoding='utf-8')
                print(f"🗺️ ROI 矢量边界已成功保存为 Shapefile: {shp_path}")

        except ImportError:
            print("\n⚠️ 警告: 缺少 geopandas 或 shapely 库，无法导出 Shapefile。")
            print("如需导出，请在终端运行: pip install geopandas shapely\n")


if __name__ == "__main__":
    METHOD_1_PATHS = [r"H:\InSAR_Mosaic_Project_ia\Results\171\1\Strip_11_vel.tif",
                      r"H:\InSAR_Mosaic_Project_ia\Results\171\1\Strip_113_vel.tif"]
    # 方法 2 的两张影像路径 [左图2, 右图2] (必须有重叠)
    # 此处假设您填写另一组影像，若无则暂时复制同一组，但无实际对比意义
    METHOD_2_PATHS = [r"H:\InSAR_Mosaic_Project_ia\Results\171\ia\Strip_11_vel.tif",
                      r"H:\InSAR_Mosaic_Project_ia\Results\171\ia\Strip_113_vel.tif"]

    CUSTOM_UNIT = "mm"
    OUTPUT_BASE_DIR = "Academic_Summary_Results"

    if any(not os.path.exists(p) for p in METHOD_1_PATHS + METHOD_2_PATHS):
        print("错误：配置文件路径不存在，请检查路径！")
    else:
        GlobalAcademicAnalyzer(METHOD_1_PATHS, METHOD_2_PATHS, CUSTOM_UNIT, OUTPUT_BASE_DIR)