# -*- coding: utf-8 -*-
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import logging
import os
import re
import multiprocessing

# =========================
# 配置与常量
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    force=True
)

NODATA_VALUE = -9999.0
OUT_DIR = r"H:\\InSAR_Mosaic_Project"

# 自动创建子目录
MODEL_DIR = os.path.join(OUT_DIR, "Models")
RESULT_DIR = os.path.join(OUT_DIR, "Results")

for d in [OUT_DIR, MODEL_DIR, RESULT_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# --- 核心控制参数 ---
MIN_PTS_THRESHOLD = 500  # 最小连接点数
MAX_ITERATIONS = 3  # 相对拟合：迭代剔除坏点的次数
SIGMA_THRESHOLD = 3.0  # 相对拟合：剔除阈值
FEATHER_PIXELS = 0  # 边缘羽化宽度


# =========================================================
# 1. 核心工具函数
# =========================================================
def reproject_to_grid(src, profile, resampling=Resampling.bilinear):
    """将影像重投影到统一格网"""
    arr = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)
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


def high_coh_mask_gpu(arr, threshold, frac, size):
    """GPU 加速的高相干掩膜提取"""
    try:
        x = cp.array(arr)
        valid = ~cp.isnan(x)
        kernel = cp.ones((size, size), dtype=cp.float32)
        count_total = convolve(valid.astype(cp.float32), kernel, mode="constant", cval=0.0)
        count_high = convolve((x > threshold).astype(cp.float32), kernel, mode="constant", cval=0.0)
        coverage = count_high / cp.maximum(count_total, 1.0)
        return cp.asnumpy((coverage >= frac) & (x >= threshold))
    except Exception:
        return np.isfinite(arr) & (arr > threshold)


def get_adaptive_mask(a1, a2):
    """自适应掩膜策略"""
    valid_data = np.isfinite(a1) & np.isfinite(a2)
    m1 = high_coh_mask_gpu(a1, 0.35, 0.4, 3)
    m2 = high_coh_mask_gpu(a2, 0.35, 0.4, 3)
    return m1 & m2 & valid_data


# =========================================================
# 2. Worker 函数 (CPU 多进程 IO)
# =========================================================
def worker_fetch_tile(args):
    """
    CPU 子进程任务：负责 IO 读取和 WarpedVRT 重投影。
    """
    (window_tuple, out_prof_clean, images_info,
     global_transform, global_crs, nodata_val) = args

    col_off, row_off, w, h = window_tuple
    window = Window(col_off, row_off, w, h)
    tile_data_stack = []

    vrt_options = {
        'resampling': Resampling.bilinear,
        'crs': global_crs,
        'transform': global_transform,
        'height': out_prof_clean['height'],
        'width': out_prof_clean['width'],
        'nodata': np.nan
    }

    w_l, w_b, w_r, w_t = rasterio.windows.bounds(window, global_transform)

    for idx, img in enumerate(images_info):
        try:
            with rasterio.open(img['vel']) as src:
                b = src.bounds
                if (b.left > w_r or b.right < w_l or b.bottom > w_t or b.top < w_b):
                    continue

            with rasterio.open(img['vel']) as src, WarpedVRT(src, **vrt_options) as vrt:
                vel_data = vrt.read(1, window=window)

            with rasterio.open(img['coh']) as src_c, WarpedVRT(src_c, **vrt_options) as vrt_c:
                coh_data = vrt_c.read(1, window=window)

            vel_data[np.isclose(vel_data, 0.0)] = np.nan
            vel_data[np.isclose(vel_data, nodata_val)] = np.nan

            if np.any(np.isfinite(vel_data)):
                coh_data = np.nan_to_num(coh_data, nan=0.0)
                tile_data_stack.append((idx, vel_data, coh_data))

        except Exception:
            continue

    return window_tuple, tile_data_stack


# =========================================================
# 3. 核心类：InSARGlobalMosaic
# =========================================================
class InSARGlobalMosaic:
    def __init__(self, image_list):
        self.images = image_list
        self.num_images = len(image_list)
        self.centers = {}

        # 6 参数模型: [Const, x, y, x^2, y^2, xy]
        self.final_corrections = np.zeros((self.num_images, 6))

        self.ref_x = 0.0
        self.ref_y = 0.0
        self.norm_scale = 100000.0  # 坐标归一化因子

        # 网络结构
        self.network_edges = []
        self.network_weights = []
        self.network_samples = []  # 存储所有有效重叠点 (x, y, diff)
        self.edge_types = []  # 1=同轨, 0=异轨

        self.track_pattern = re.compile(r'vel_\d+_(\d+)_\d+')
        self._init_geometry_info()

    def _get_track_id(self, idx):
        filename = os.path.basename(self.images[idx]['vel'])
        match = self.track_pattern.search(filename)
        return match.group(1) if match else "U"

    def _init_geometry_info(self):
        self.centers = {}
        self.bounds_cache = {}
        xs, ys = [], []
        logging.info("Initializing geometry info (caching bounds)...")

        for idx, img in enumerate(self.images):
            with rasterio.open(img['vel']) as src:
                self.bounds_cache[idx] = src.bounds
                cx, cy = (src.bounds.left + src.bounds.right) / 2, (src.bounds.top + src.bounds.bottom) / 2
                xs.append(cx);
                ys.append(cy)
                self.centers[idx] = (cx, cy)

        self.ref_x = np.mean(xs)
        self.ref_y = np.mean(ys)
        logging.info(f"Reference Point: ({self.ref_x:.2f}, {self.ref_y:.2f})")

    def _eval_poly(self, params, px, py):
        """辅助函数：计算6参数多项式值"""
        return (params[0] +
                params[1] * px +
                params[2] * py +
                params[3] * px ** 2 +
                params[4] * py ** 2 +
                params[5] * px * py)

    # ------------------------------------------------------------------
    # 修改点 1: 相对拟合，保留所有有效点 (不随机抽样)
    # ------------------------------------------------------------------
    def _calculate_robust_params(self, idx1, idx2):
        img1, img2 = self.images[idx1], self.images[idx2]
        track1 = self._get_track_id(idx1)
        track2 = self._get_track_id(idx2)
        is_same_track = (track1 == track2) and (track1 != "U")

        # 1. 几何求交
        with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2:
            xl = max(src1.bounds.left, src2.bounds.left)
            xr = min(src1.bounds.right, src2.bounds.right)
            yb = max(src1.bounds.bottom, src2.bounds.bottom)
            yt = min(src1.bounds.top, src2.bounds.top)
            # 返回值：(params, weight, samples_tuple)
            if xr <= xl or yt <= yb: return None, 0, None

        # 2. 读取数据
        with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2, \
                rasterio.open(img1['coh']) as c_src1, rasterio.open(img2['coh']) as c_src2:

            width = int((xr - xl) / src1.res[0])
            height = int((yt - yb) / src1.res[1])
            if width < 5 or height < 5: return None, 0, None

            transform = from_origin(xl, yt, src1.res[0], src1.res[1])
            prof = {'height': height, 'width': width, 'transform': transform, 'crs': src1.crs}

            try:
                v1_arr = reproject_to_grid(src1, prof)
                v2_arr = reproject_to_grid(src2, prof)
                c1_arr = np.nan_to_num(reproject_to_grid(c_src1, prof))
                c2_arr = np.nan_to_num(reproject_to_grid(c_src2, prof))
            except Exception:
                return None, 0, None

            v1_arr[np.isclose(v1_arr, 0.0) | np.isclose(v1_arr, NODATA_VALUE)] = np.nan
            v2_arr[np.isclose(v2_arr, 0.0) | np.isclose(v2_arr, NODATA_VALUE)] = np.nan

            mask_valid = get_adaptive_mask(c1_arr, c2_arr)
            if int(np.sum(mask_valid)) < MIN_PTS_THRESHOLD: return None, 0, None

            # 3. 准备数据
            idx_flat = np.flatnonzero(mask_valid)
            v1 = v1_arr.ravel()[idx_flat].astype(np.float64)
            v2 = v2_arr.ravel()[idx_flat].astype(np.float64)
            w = (c1_arr.ravel()[idx_flat] * c2_arr.ravel()[idx_flat]) ** 2

            rows, cols = np.indices(v1_arr.shape)
            px_all, py_all = transform * (cols, rows)

            # 归一化坐标
            px_pts = (px_all.ravel()[idx_flat] - self.ref_x) / self.norm_scale
            py_pts = (py_all.ravel()[idx_flat] - self.ref_y) / self.norm_scale

            diff = v1 - v2
            clean_mask = (np.isfinite(diff) & np.isfinite(w) &
                          np.isfinite(px_pts) & np.isfinite(py_pts))

            diff = diff[clean_mask]
            w = w[clean_mask]
            px_pts = px_pts[clean_mask]
            py_pts = py_pts[clean_mask]

            if len(diff) < MIN_PTS_THRESHOLD: return None, 0, None

            current_inliers = np.ones(len(diff), dtype=bool)
            sp_params = np.zeros(6)
            final_rmse = 0.0

            try:
                # 迭代剔除离群值 (获取纯净的重叠区点集)
                for iteration in range(MAX_ITERATIONS):
                    if np.sum(current_inliers) < 100: return None, 0, None
                    z_sub = diff[current_inliers]
                    px_sub = px_pts[current_inliers]
                    py_sub = py_pts[current_inliers]
                    W_sqrt = np.sqrt(w[current_inliers])[:, None]

                    if is_same_track:
                        A_mat = np.column_stack((np.ones_like(z_sub), px_sub, py_sub))
                        C, _, _, _ = np.linalg.lstsq(A_mat * W_sqrt, z_sub[:, None] * W_sqrt, rcond=None)
                        res = C.flatten()
                        sp_params[0:3] = res;
                        sp_params[3:] = 0.0
                        model_val = res[0] + res[1] * px_sub + res[2] * py_sub
                    else:
                        A_mat = np.column_stack(
                            (np.ones_like(z_sub), px_sub, py_sub, px_sub ** 2, py_sub ** 2, px_sub * py_sub))
                        C, _, _, _ = np.linalg.lstsq(A_mat * W_sqrt, z_sub[:, None] * W_sqrt, rcond=None)
                        sp_params = C.flatten()
                        model_val = (sp_params[0] + sp_params[1] * px_sub + sp_params[2] * py_sub +
                                     sp_params[3] * px_sub ** 2 + sp_params[4] * py_sub ** 2 + sp_params[
                                         5] * px_sub * py_sub)

                    resid = np.abs(z_sub - model_val)
                    sigma = np.std(resid)
                    if sigma < 1e-9: sigma = 1e-9
                    threshold = SIGMA_THRESHOLD * sigma
                    keep_mask = (resid < threshold)
                    current_inliers[current_inliers] = keep_mask
                    final_rmse = sigma

                # === 关键修改：保存所有有效内点 (Inliers) ===
                # 转为 float32 以节省内存，因为这些点只用于计算 RMSE
                px_save = px_pts[current_inliers].astype(np.float32)
                py_save = py_pts[current_inliers].astype(np.float32)
                diff_save = diff[current_inliers].astype(np.float32)

                # samples: (x, y, 原始观测差值)
                samples = (px_save, py_save, diff_save)

                n_points = np.sum(current_inliers)
                safe_sigma = max(final_rmse, 1e-6)
                weight_val = n_points / (safe_sigma ** 2)

                return sp_params, weight_val, samples

            except Exception as e:
                logging.error(f"Fit error: {e}")
                return None, 0, None

    # ------------------------------------------------------------------
    # 修改点 2: 缓存所有采样点
    # ------------------------------------------------------------------
    def build_network_topology(self):
        logging.info("Step 1: Building Network Topology (Caching FULL overlap points)...")
        self.network_edges = []
        self.network_weights = []
        self.network_samples = []  # 新增：存所有点
        self.edge_types = []

        count = 0

        for i in range(self.num_images):
            track_i = self._get_track_id(i)
            for j in range(i + 1, self.num_images):
                track_j = self._get_track_id(j)
                l1, b1, r1, t1 = self.bounds_cache[i]
                l2, b2, r2, t2 = self.bounds_cache[j]

                if not (l1 > r2 or r1 < l2 or b1 > t2 or t1 < b2):
                    # 获取 参数, 初始权重, 和全量样本
                    params_diff, _, samples = self._calculate_robust_params(i, j)

                    if params_diff is not None:
                        self.network_edges.append((i, j, params_diff))
                        self.network_samples.append(samples)  # <--- 存入列表

                        if (track_i == track_j) and (track_i != "U"):
                            self.network_weights.append(100.0)  # 同轨初始高权
                            self.edge_types.append(1)
                        else:
                            self.network_weights.append(1.0)
                            self.edge_types.append(0)
                        count += 1

        logging.info(f"Topology built. Total Edges: {count}. All overlap points cached.")

    # ------------------------------------------------------------------
    # ALS 求解器 (保持不变)
    # ------------------------------------------------------------------
    def solve_with_current_weights(self):
        if not self.network_edges: return None

        # 锚点权重
        W_ANCHOR = 100.0  # 稍微降低一点，避免淹没观测值

        num_vars_per_img = 6
        num_unk = self.num_images * num_vars_per_img

        # === 改用 Dense Matrix (稠密矩阵) ===
        # 预估行数：边数*6 + 同轨约束*3 + 锚点*6
        # 直接用 list 收集然后 vstack，或者估算最大行数
        rows_data = []  # 存 (A_row, b_val)

        # A: 观测方程
        for idx, (i, j, p) in enumerate(self.network_edges):
            curr_w = self.network_weights[idx]
            # 限制权重下限，防止除以0
            if curr_w < 1e-6: curr_w = 1e-6

            # 【关键】对权重开根号，因为 lstsq 解的是 sum( (Ax-b)^2 )
            # 我们希望 minimize sum( w * (Ax-b)^2 ) => minimize sum( (sqrt(w)Ax - sqrt(w)b)^2 )
            w_sqrt = np.sqrt(curr_w)

            # 6个参数
            for k in range(num_vars_per_img):
                # 构建一行: [..., coeff_i, ..., coeff_j, ...] = target
                # 这里的 coeff 只是 1.0 或 -1.0，但要乘以权重

                # 对应方程： Param_i[k] - Param_j[k] = p[k]
                row = np.zeros(num_unk, dtype=np.float64)

                start_i = i * num_vars_per_img
                start_j = j * num_vars_per_img

                row[start_i + k] = 1.0 * w_sqrt
                row[start_j + k] = -1.0 * w_sqrt

                target = p[k] * w_sqrt
                rows_data.append((row, target))

        # B: 同轨锁定 (Same Track Locking)
        # 只要是同轨，强迫高次项系数相同
        w_lock = np.sqrt(10.0)  # 适当的锁定权重
        for i in range(self.num_images):
            for j in range(i + 1, self.num_images):
                t1 = self._get_track_id(i)
                t2 = self._get_track_id(j)
                if t1 == t2 and t1 != "U":
                    # 仅锁定 x^2, y^2, xy (k=3,4,5)
                    for k in [3, 4, 5]:
                        row = np.zeros(num_unk, dtype=np.float64)
                        row[i * 6 + k] = 1.0 * w_lock
                        row[j * 6 + k] = -1.0 * w_lock
                        rows_data.append((row, 0.0))

        # C: 锚点 (Anchor) - 固定最靠近中心的影像
        dists = [np.hypot(self.centers[i][0] - self.ref_x, self.centers[i][1] - self.ref_y) for i in
                 range(self.num_images)]
        anchor_idx = np.argmin(dists)
        w_anchor_sqrt = np.sqrt(W_ANCHOR)

        for k in range(num_vars_per_img):
            row = np.zeros(num_unk, dtype=np.float64)
            row[anchor_idx * 6 + k] = 1.0 * w_anchor_sqrt
            # 目标值为0，即强迫锚点参数为0 (不校正)
            rows_data.append((row, 0.0))

        # 转换为矩阵
        num_rows = len(rows_data)
        A = np.zeros((num_rows, num_unk), dtype=np.float64)
        B = np.zeros(num_rows, dtype=np.float64)

        for r_idx, (r_vec, b_val) in enumerate(rows_data):
            A[r_idx, :] = r_vec
            B[r_idx] = b_val

        # === 使用 numpy 直接求解 (SVD) ===
        # rcond=None 让 numpy 自动处理奇异值，非常鲁棒
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
            return x
        except Exception as e:
            logging.error(f"Matrix solution failed: {e}")
            return None

    # ------------------------------------------------------------------
    # 修改点 3: 全网平差 (基于全量像素 RMSE 定权)
    # ------------------------------------------------------------------
    def solve_global_network_igg(self):
        logging.info("Step 2: Starting Global Forced Correction (Dense Matrix & Stable Weights)...")

        if not self.network_edges:
            return

        MAX_LOOPS = 15
        TARGET_RMSE = 0.002  # 2mm
        FIXED_WEIGHT = 100.0

        prev_models = np.zeros((self.num_images, 6))

        for iter_idx in range(MAX_LOOPS):
            # 1. ALS 求解
            sol_vec = self.solve_with_current_weights()
            if sol_vec is None:
                logging.error("Solver failed.")
                break

            current_models = np.zeros((self.num_images, 6))
            for i in range(self.num_images):
                current_models[i, :] = sol_vec[i * 6: (i + 1) * 6]
            self.final_corrections = current_models

            # 2. 计算 RMSE
            residuals_rmse_list = []
            max_st_rmse = 0.0
            max_ct_rmse = 0.0

            for idx, (i, j, _) in enumerate(self.network_edges):
                px_s, py_s, obs_diff_s = self.network_samples[idx]
                corr_i = self._eval_poly(current_models[i], px_s, py_s)
                corr_j = self._eval_poly(current_models[j], px_s, py_s)

                residual_gap = obs_diff_s - (corr_i - corr_j)
                rmse = np.sqrt(np.mean(residual_gap ** 2))
                residuals_rmse_list.append(rmse)

                if self.edge_types[idx] == 1:
                    max_st_rmse = max(max_st_rmse, rmse)
                else:
                    max_ct_rmse = max(max_ct_rmse, rmse)

            # 打印当前状态和平均权重，检查权重是否在变
            avg_weight = np.mean(self.network_weights)
            logging.info(
                f"--- Iter {iter_idx + 1}: Max ST RMSE={max_st_rmse * 1000:.2f}mm, Max CT RMSE={max_ct_rmse * 1000:.2f}mm, AvgW={avg_weight:.1f}")

            # 3. 收敛判断
            if max_st_rmse < TARGET_RMSE and iter_idx > 2:
                logging.info(f"   >>> Converged. Same-track aligned.")
                break

            prev_models = current_models.copy()

            # 4. 权重更新 (限制上限，防止病态)
            boost_count = 0
            for idx in range(len(self.network_edges)):
                rmse = residuals_rmse_list[idx]
                old_w = self.network_weights[idx]
                is_same_track = (self.edge_types[idx] == 1)

                if is_same_track:
                    if rmse > TARGET_RMSE:
                        # 稍微温和的倍增
                        new_w = old_w * 3.0
                        # 【重要】封顶值改小，防止数值计算崩溃
                        if new_w > 100000.0: new_w = 100000.0
                        self.network_weights[idx] = new_w
                        boost_count += 1
                else:
                    # 异轨降权
                    new_w = 0.0001 / (rmse ** 2 + 1e-8)
                    if new_w > 10.0: new_w = 10.0
                    if new_w < 0.01: new_w = 0.01  # 别让它变成0
                    self.network_weights[idx] = new_w

            logging.info(f"   Action -> Boosted {boost_count} Same-Track edges.")

        logging.info("Correction Finished.")

    def save_model(self, filepath):
        np.save(filepath, self.final_corrections)
        logging.info(f"Correction model saved to: {filepath}")

    # ------------------------------------------------------------------
    # 混合并行拼接 (Pipeline: CPU Fetch -> GPU Compute)
    # ------------------------------------------------------------------
    def generate_mosaic_gpu_tiled(self, output_path_vel, output_path_coh=None, block_size=2048, num_workers=6):
        logging.info(f"Step 3: Starting Hybrid GPU Mosaic (Workers: {num_workers}, Block: {block_size})...")
        mempool = cp.get_default_memory_pool()

        # 1. 全局范围
        bounds = []
        res, crs = None, None
        for img in self.images:
            with rasterio.open(img['vel']) as src:
                bounds.append(src.bounds)
                if res is None: res = src.res; crs = src.crs

        xmin = min(b.left for b in bounds);
        xmax = max(b.right for b in bounds)
        ymin = min(b.bottom for b in bounds);
        ymax = max(b.top for b in bounds)
        width = int(np.ceil((xmax - xmin) / res[0]))
        height = int(np.ceil((ymax - ymin) / res[1]))
        transform = from_origin(xmin, ymax, res[0], res[1])

        out_prof_clean = {'height': height, 'width': width, 'transform': transform, 'crs': crs}
        write_prof = {
            'driver': 'GTiff', 'height': height, 'width': width, 'count': 1,
            'dtype': 'float32', 'crs': crs, 'transform': transform, 'nodata': NODATA_VALUE,
            'compress': 'lzw', 'tiled': True, 'blockxsize': 256, 'blockysize': 256
        }

        # 2. 生成任务列表
        tasks = []
        for row_off in range(0, height, block_size):
            for col_off in range(0, width, block_size):
                w = min(block_size, width - col_off)
                h = min(block_size, height - row_off)
                tasks.append(((col_off, row_off, w, h), out_prof_clean, self.images, transform, crs, NODATA_VALUE))

        logging.info(f"Total Tiles: {len(tasks)} | Grid: {width}x{height}")

        # 3. 启动流水线
        with rasterio.open(output_path_vel, "w", **write_prof) as dst_v:
            dst_coh = rasterio.open(output_path_coh, "w", **write_prof) if output_path_coh else None

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(worker_fetch_tile, task): task for task in tasks}
                pbar = tqdm(total=len(tasks), desc="Pipeline Mosaic")

                for future in as_completed(futures):
                    try:
                        window_tuple, tile_data_stack = future.result()
                        col_off, row_off, w, h = window_tuple
                        window = Window(col_off, row_off, w, h)

                        if not tile_data_stack:
                            pbar.update(1)
                            continue

                        # GPU 计算
                        tile_transform = rasterio.windows.transform(window, transform)
                        sum_v_gpu = cp.zeros((h, w), dtype=cp.float32)
                        sum_c_gpu = cp.zeros((h, w), dtype=cp.float32)
                        sum_w_gpu = cp.zeros((h, w), dtype=cp.float32)

                        rows_gpu, cols_gpu = cp.indices((h, w), dtype=cp.float32)
                        a, b, c_tx = tile_transform.a, tile_transform.b, tile_transform.c
                        d, e, f_ty = tile_transform.d, tile_transform.e, tile_transform.f

                        px_gpu = a * cols_gpu + b * rows_gpu + c_tx
                        py_gpu = d * cols_gpu + e * rows_gpu + f_ty

                        px_norm_gpu = (px_gpu - self.ref_x) / self.norm_scale
                        py_norm_gpu = (py_gpu - self.ref_y) / self.norm_scale
                        del rows_gpu, cols_gpu, px_gpu, py_gpu

                        has_valid_pixel = False

                        for idx, vel_data, coh_data in tile_data_stack:
                            data_gpu = cp.asarray(vel_data)
                            coh_gpu = cp.asarray(coh_data)
                            params_gpu = cp.asarray(self.final_corrections[idx])

                            mask_gpu = (~cp.isnan(data_gpu)) & (coh_gpu > 0.05)

                            if cp.sum(mask_gpu) > 0:
                                has_valid_pixel = True
                                correction = (params_gpu[0] +
                                              params_gpu[1] * px_norm_gpu +
                                              params_gpu[2] * py_norm_gpu +
                                              params_gpu[3] * (px_norm_gpu ** 2) +
                                              params_gpu[4] * (py_norm_gpu ** 2) +
                                              params_gpu[5] * (px_norm_gpu * py_norm_gpu))

                                val_corrected = data_gpu - correction

                                # 羽化
                                mask_cpu = cp.asnumpy(mask_gpu)
                                dist_cpu = distance_transform_edt(mask_cpu)
                                dist_gpu = cp.asarray(dist_cpu, dtype=cp.float32)
                                dist_norm_gpu = cp.clip(dist_gpu / FEATHER_PIXELS, 0.0,
                                                        1.0) if FEATHER_PIXELS > 0 else cp.ones_like(dist_gpu)
                                geo_weight_gpu = 0.5 * (1.0 - cp.cos(dist_norm_gpu * cp.pi))
                                pixel_w_gpu = geo_weight_gpu * cp.sqrt(coh_gpu + 0.01)

                                sum_v_gpu = cp.where(mask_gpu, sum_v_gpu + val_corrected * pixel_w_gpu, sum_v_gpu)
                                sum_c_gpu = cp.where(mask_gpu, sum_c_gpu + coh_gpu * pixel_w_gpu, sum_c_gpu)
                                sum_w_gpu = cp.where(mask_gpu, sum_w_gpu + pixel_w_gpu, sum_w_gpu)

                                del mask_cpu, dist_cpu, dist_gpu

                            del data_gpu, coh_gpu, mask_gpu, params_gpu

                        if has_valid_pixel:
                            final_v_gpu = cp.full((h, w), NODATA_VALUE, dtype=cp.float32)
                            final_c_gpu = cp.full((h, w), 0, dtype=cp.float32)

                            valid_gpu = sum_w_gpu > 1e-4
                            final_v_gpu[valid_gpu] = sum_v_gpu[valid_gpu] / sum_w_gpu[valid_gpu]
                            final_c_gpu[valid_gpu] = sum_c_gpu[valid_gpu] / sum_w_gpu[valid_gpu]

                            dst_v.write(cp.asnumpy(final_v_gpu), 1, window=window)
                            if dst_coh: dst_coh.write(cp.asnumpy(final_c_gpu), 1, window=window)

                        del sum_v_gpu, sum_c_gpu, sum_w_gpu, px_norm_gpu, py_norm_gpu
                        mempool.free_all_blocks()

                    except Exception as e:
                        logging.error(f"Tile processing error: {e}")
                    pbar.update(1)

                pbar.close()
            if dst_coh: dst_coh.close()
        logging.info(f"SUCCESS: Pipeline Mosaic Saved to {output_path_vel}")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    target_year = 19

    # inputs = [
    #     {'id': 0, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_126\vel_{target_year}_113_126_m.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_113_126\coh_{target_year}_113_126.tif"},
    #     {'id': 1, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_121\vel_{target_year}_113_121_m.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_113_121\coh_{target_year}_113_121.tif"},
    #     {'id': 2, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_116\vel_{target_year}_113_116_m.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_113_116\coh_{target_year}_113_116.tif"},
    #     {'id': 3, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_111\vel_{target_year}_113_111_m.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_113_111\coh_{target_year}_113_111.tif"},
    #     {'id': 4, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_121\vel_{target_year}_11_121_m.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_11_121\coh_{target_year}_11_121.tif"},
    #     {'id': 5, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_116\vel_{target_year}_11_116_m.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_11_116\coh_{target_year}_11_116.tif"},
    #     {'id': 6, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_111\vel_{target_year}_11_111_m.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_11_111\coh_{target_year}_11_111.tif"},
    #     {'id': 7, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_127\vel_{target_year}_40_127.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_40_127\coh_{target_year}_40_127.tif"},
    #     {'id': 8, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_122\vel_{target_year}_40_122_m.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_40_122\coh_{target_year}_40_122.tif"},
    #     {'id': 9, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_117\vel_{target_year}_40_117.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_40_117\coh_{target_year}_40_117.tif"},
    # ]

    inputs = [
        {'id': 7, 'vel': fr"H:\InSAR_Mosaic_Project\Results\vel_113_111.tif", 'coh': fr"H:\InSAR_Mosaic_Project\Results\vel_113_111.tif.tif"},
        {'id': 8, 'vel': fr"H:\InSAR_Mosaic_Project\Results\vel_11_111.tif", 'coh': fr"H:\InSAR_Mosaic_Project\Results\coh_11_111.tif"},
        {'id': 9, 'vel': fr"H:\InSAR_Mosaic_Project\Results\vel_40_111.tif", 'coh': fr"H:\InSAR_Mosaic_Project\Results\coh_40_111.tif"},
    ]

    logging.info("====== Starting InSAR Global IGG Robust Mosaic ======")
    logging.info(f"Total Images: {len(inputs)}")

    app = InSARGlobalMosaic(inputs)

    # 1. 建立网络拓扑 (缓存所有有效点)
    app.build_network_topology()

    # 2. IGG 迭代求解 (基于真实 RMSE 迭代定权)
    app.solve_global_network_igg()

    # 3. 保存模型
    app.save_model(os.path.join(MODEL_DIR, "Global_Model.npy"))

    out_vel = os.path.join(RESULT_DIR, "Final_Global_Mosaic_Vel.tif")
    out_coh = os.path.join(RESULT_DIR, "Final_Global_Mosaic_Coh.tif")

    # 4. 混合并行拼接
    cpu_cores = os.cpu_count()
    workers = max(1, cpu_cores - 2) if cpu_cores else 4
    app.generate_mosaic_gpu_tiled(out_vel, out_coh, block_size=2048, num_workers=workers)

    logging.info("ALL DONE.")