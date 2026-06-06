# -*- coding: utf-8 -*-
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve, percentile_filter, distance_transform_edt
import logging
import os
import re
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order

# =========================
# 配置与常量
# =========================
NODATA_VALUE = -9999.0
eps_w = 1e-6
gamma = 2.0
OUT_DIR = r"H:\\InSAR_Mosaic_Project4"  # 输出总目录

# 自动创建子目录
MODEL_DIR = os.path.join(OUT_DIR, "Models")
RESULT_DIR = os.path.join(OUT_DIR, "Results")
for d in [OUT_DIR, MODEL_DIR, RESULT_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# --- 核心防御参数 ---
MAX_CORRECTION = 0.2
MIN_SPREAD_RATIO = 0.05
TARGET_MIN_POINTS = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# =========================================================
# 1. 核心数学工具库 (GPU 加速版)
# =========================================================

def reproject_to_grid(src, profile, resampling=Resampling.bilinear):
    """
    CPU端进行重投影 (Rasterio依赖CPU)，但返回numpy数组以便后续上传GPU
    """
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


def write_with_nodata(arr, filename, out_profile):
    """将数据写入磁盘 (CPU)"""
    # 如果 arr 还在 GPU 上，先传回 CPU
    if isinstance(arr, cp.ndarray):
        arr = cp.asnumpy(arr)

    arr_out = np.where(np.isnan(arr), NODATA_VALUE, arr).astype(np.float32)
    with rasterio.open(filename, "w", **out_profile) as dst:
        dst.write(arr_out, 1)


def high_coh_mask_gpu(arr, threshold, frac, size):
    """完全在 GPU 上计算掩膜"""
    # 输入如果是 numpy，转为 cupy；如果是 cupy 则保持
    x = cp.asarray(arr)
    valid = ~cp.isnan(x)
    kernel = cp.ones((size, size), dtype=cp.float32)
    count_total = convolve(valid.astype(cp.float32), kernel, mode="constant", cval=0.0)
    count_high = convolve((x > threshold).astype(cp.float32), kernel, mode="constant", cval=0.0)
    coverage = count_high / cp.maximum(count_total, 1.0)

    # 填充 nan 以便计算中值
    x_filled = cp.where(valid, x, threshold - 1e-6)
    med = percentile_filter(x_filled, percentile=50, size=size, mode="nearest")

    mask = (coverage >= frac) & (med >= threshold) & (x >= threshold)
    return mask  # 返回 GPU array


def normalize_t_and_b_gpu(t, b1, b2, c1=0.0, c2=0.0, wsum=None):
    """GPU 版归一化"""
    if wsum is not None:
        t_mean = cp.average(t, weights=wsum)
        t_centered = t - t_mean
        t_std = cp.sqrt(cp.average(t_centered ** 2, weights=wsum))
    else:
        t_mean = t.mean()
        t_centered = t - t_mean
        t_std = cp.sqrt(cp.mean(t_centered ** 2))

    if t_std < 1e-12: t_std = 1.0

    t_new = t_centered / t_std
    b1_new = b1 * t_std
    b2_new = b2 * t_std
    c1_new = c1 * (t_std ** 2)
    c2_new = c2 * (t_std ** 2)

    if b1_new < 0:
        t_new = -t_new
        b1_new = -b1_new
        b2_new = -b2_new
        c1_new = -c1_new
        c2_new = -c2_new

    return t_new, b1_new, b2_new, c1_new, c2_new


def joint_linreg_gpu(v1, v2, t, w1, w2, wsum, lambda_delta=1e-2):
    """GPU 版联合线性回归"""
    # 构建矩阵 X (N, 2)
    ones = cp.ones_like(t)
    X = cp.column_stack([ones, t])

    # 矩阵运算全部在 GPU
    # X.T @ (w * X)
    Xt = X.T

    # A11
    w1_X = w1[:, None] * X
    A11 = Xt @ w1_X

    # A22
    w2_X = w2[:, None] * X
    A22 = Xt @ w2_X

    # Vectors b
    b1_vec = Xt @ (w1 * v1)
    b2_vec = Xt @ (w2 * v2)

    # Regularization parts
    Axx = Xt @ (wsum[:, None] * X)

    A11 += lambda_delta * Axx
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx
    A21 = A12.T

    b1_vec += lambda_delta * Xt @ (wsum * v2)
    b2_vec += lambda_delta * Xt @ (wsum * v1)

    # Block matrix construction
    # [[A11, A12],
    #  [A21, A22]]
    r1 = cp.hstack([A11, A12])
    r2 = cp.hstack([A21, A22])
    A = cp.vstack([r1, r2])

    b = cp.concatenate([b1_vec, b2_vec])

    # Solve on GPU
    # Add tiny jitter for stability
    A += 1e-8 * cp.eye(4)
    params = cp.linalg.solve(A, b)

    # Return as numpy for consistency with lightweight logic, or keep as cupy?
    # Keeping logic light: return numpy scalar params
    return cp.asnumpy(params)


def joint_quadreg_gpu(v1, v2, t, w1, w2, wsum, lambda_delta=1e-2):
    """GPU 版联合二次回归"""
    X = cp.column_stack([cp.ones_like(t), t, t ** 2])
    Xt = X.T

    A11 = Xt @ (w1[:, None] * X)
    A22 = Xt @ (w2[:, None] * X)
    b1_vec = Xt @ (w1 * v1)
    b2_vec = Xt @ (w2 * v2)

    Axx = Xt @ (wsum[:, None] * X)

    A11 += lambda_delta * Axx
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx
    A21 = A12.T

    b1_vec += lambda_delta * Xt @ (wsum * v2)
    b2_vec += lambda_delta * Xt @ (wsum * v1)

    r1 = cp.hstack([A11, A12])
    r2 = cp.hstack([A21, A22])
    A = cp.vstack([r1, r2])
    b = cp.concatenate([b1_vec, b2_vec])

    A = 0.5 * (A + A.T)
    reg = 1e-8 * cp.trace(A) / max(A.shape[0], 1)
    A += reg * cp.eye(A.shape[0])

    params = cp.linalg.solve(A, b)
    return cp.asnumpy(params)


def joint_constreg_gpu(v1, v2, w1, w2, wsum, lambda_delta=1e-2):
    """GPU 版常数回归"""
    X = cp.ones((len(v1), 1))
    Xt = X.T

    A11 = Xt @ (w1[:, None] * X)
    A22 = Xt @ (w2[:, None] * X)
    b1_vec = Xt @ (w1 * v1)
    b2_vec = Xt @ (w2 * v2)

    Axx = Xt @ (wsum[:, None] * X)

    A11 += lambda_delta * Axx
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx
    A21 = A12.T

    b1_vec += lambda_delta * Xt @ (wsum * v2)
    b2_vec += lambda_delta * Xt @ (wsum * v1)

    r1 = cp.hstack([A11, A12])
    r2 = cp.hstack([A21, A22])
    A = cp.vstack([r1, r2])
    b = cp.concatenate([b1_vec, b2_vec])

    params = cp.linalg.solve(A + 1e-8 * cp.eye(2), b)
    return cp.asnumpy(params)


def wrmse_gpu(res, w):
    """GPU Weighted RMSE"""
    res_valid = res[cp.isfinite(res)]
    w_valid = w[cp.isfinite(res)]
    return cp.asnumpy(cp.sqrt(cp.sum(w_valid * res_valid ** 2) / (cp.sum(w_valid) + 1e-12)))


def evaluate_model_shared_t_gpu(v1, v2, w1, w2, wsum, t, lambda_delta=1e-2, model="linear"):
    # v1, v2, w1... are GPU arrays
    if model == "const":
        params = joint_constreg_gpu(v1, v2, w1, w2, wsum, lambda_delta)
        a1_c, a2_c = params[0], params[1]
        fit1 = cp.full_like(v1, a1_c)
        fit2 = cp.full_like(v2, a2_c)
    elif model == "linear":
        params = joint_linreg_gpu(v1, v2, t, w1, w2, wsum, lambda_delta)
        a1_l, b1_l, a2_l, b2_l = params
        fit1 = a1_l + b1_l * t
        fit2 = a2_l + b2_l * t
    elif model == "quadratic":
        params = joint_quadreg_gpu(v1, v2, t, w1, w2, wsum, lambda_delta)
        a1_q, b1_q, c1_q, a2_q, b2_q, c2_q = params
        fit1 = a1_q + b1_q * t + c1_q * t ** 2
        fit2 = a2_q + b2_q * t + c2_q * t ** 2
    else:
        raise ValueError("Unknown model")

    r1, r2 = v1 - fit1, v2 - fit2
    diff_adj = fit1 - fit2

    return {
        "model": model, "lambda": lambda_delta,
        "wrmse1": wrmse_gpu(r1, w1),
        "wrmse2": wrmse_gpu(r2, w2),
        "wrmse_diff": wrmse_gpu(diff_adj, wsum),
        "bias_diff": cp.asnumpy(cp.average(diff_adj, weights=wsum))
    }


def evaluate_all_models_gpu(v1, v2, t, w1, w2, wsum, lambda_delta):
    results = []
    for model in ["const", "linear", "quadratic"]:
        scores = evaluate_model_shared_t_gpu(v1, v2, w1, w2, wsum, t, lambda_delta=lambda_delta, model=model)
        results.append({"lambda": lambda_delta, "model": model,
                        "metrics": (scores["wrmse_diff"], scores["wrmse1"], scores["wrmse2"]), "scores": scores})
    return results


def pareto_frontier(points):
    frontier = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j:
                if all(qm <= pm for qm, pm in zip(q["metrics"], p["metrics"])) and \
                        any(qm < pm for qm, pm in zip(q["metrics"], p["metrics"])):
                    dominated = True;
                    break
        if not dominated: frontier.append(p)
    return frontier


def ideal_point_choice(frontier):
    best = None
    best_val = np.inf
    for p in frontier:
        d, r1, r2 = p["metrics"]
        val = np.sqrt(d ** 2 + r1 ** 2 + r2 ** 2)
        if val < best_val: best_val = val; best = p
    return best


def get_adaptive_mask_gpu(c1_gpu, c2_gpu, level):
    if level == 0:
        m1 = high_coh_mask_gpu(c1_gpu, 0.75, 0.65, 3)
        m2 = high_coh_mask_gpu(c2_gpu, 0.75, 0.65, 3)
        return m1 & m2, "Strategy: Strict"
    elif level == 1:
        m1 = (c1_gpu >= 0.40)
        m2 = (c2_gpu >= 0.40)
        return m1 & m2, "Strategy: Loose"
    else:
        m1 = (c1_gpu >= 0.10) & ~cp.isnan(c1_gpu)
        m2 = (c2_gpu >= 0.10) & ~cp.isnan(c2_gpu)
        return m1 & m2, "Strategy: Fallback"


# =========================================================
# 2. 核心类：InSARMSTMosaic (GPU Accelerated)
# =========================================================
class InSARMSTMosaic:
    def __init__(self, image_list, fitting_order=1):
        self.images = image_list
        self.num_images = len(image_list)
        self.fitting_order = fitting_order
        self.centers = {}

        self.final_corrections = np.zeros((self.num_images, 6))

        self.ref_x = 0.0
        self.ref_y = 0.0
        self.norm_scale = 1.0
        self._init_geometry_info()

    def _init_geometry_info(self):
        xs, ys = [], []
        logging.info("Initializing geometry info...")
        for idx, img in enumerate(self.images):
            # Geometry info is small, can keep on CPU/Disk read
            with rasterio.open(img['vel']) as src:
                l, b, r, t = src.bounds
                cx, cy = (l + r) / 2, (t + b) / 2
                xs.append(cx)
                ys.append(cy)
                self.centers[idx] = (cx, cy)
        self.ref_x = np.mean(xs)
        self.ref_y = np.mean(ys)
        max_dist = 0.0
        for idx in self.centers:
            d = np.sqrt((self.centers[idx][0] - self.ref_x) ** 2 + (self.centers[idx][1] - self.ref_y) ** 2)
            if d > max_dist: max_dist = d
        self.norm_scale = max_dist if max_dist > 1000 else 10000.0

    # =========================================================
    # 核心计算函数：GPU 加速版
    # =========================================================
    def _calculate_robust_params(self, idx1, idx2):
        img1, img2 = self.images[idx1], self.images[idx2]

        name1 = os.path.basename(img1['vel']).split('.')[0]
        name2 = os.path.basename(img2['vel']).split('.')[0]
        pair_tag = f"[{idx1}:{name1} vs {idx2}:{name2}]"

        # 1. 几何重叠检查 (CPU快速检查)
        with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2:
            xl = max(src1.bounds.left, src2.bounds.left)
            xr = min(src1.bounds.right, src2.bounds.right)
            yb = max(src1.bounds.bottom, src2.bounds.bottom)
            yt = min(src1.bounds.top, src2.bounds.top)
            if xr <= xl or yt <= yb:
                return None, 0

            area1 = (src1.bounds.right - src1.bounds.left) * (src1.bounds.top - src1.bounds.bottom)
            area2 = (src2.bounds.right - src2.bounds.left) * (src2.bounds.top - src2.bounds.bottom)
            overlap_ratio = ((xr - xl) * (yt - yb)) / min(area1, area2)
            if overlap_ratio < 0.03:
                return None, 0

        # 2. 读取数据 (IO在CPU, 读取后立即转GPU)
        with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2, \
                rasterio.open(img1['coh']) as c_src1, rasterio.open(img2['coh']) as c_src2:

            width = int((xr - xl) / src1.res[0])
            height = int((yt - yb) / src1.res[1])
            if width < 10 or height < 10: return None, 0
            transform = from_origin(xl, yt, src1.res[0], src1.res[1])
            prof = {'height': height, 'width': width, 'transform': transform, 'crs': src1.crs}

            # Reproject (CPU)
            v1_arr_cpu = reproject_to_grid(src1, prof, resampling=Resampling.bilinear)
            v2_arr_cpu = reproject_to_grid(src2, prof, resampling=Resampling.bilinear)
            c1_arr_cpu = reproject_to_grid(c_src1, prof)
            c2_arr_cpu = reproject_to_grid(c_src2, prof)

            # Transfer to GPU
            v1_arr = cp.asarray(v1_arr_cpu)
            v2_arr = cp.asarray(v2_arr_cpu)
            c1_arr = cp.nan_to_num(cp.asarray(c1_arr_cpu))
            c2_arr = cp.nan_to_num(cp.asarray(c2_arr_cpu))

            # 清理 CPU 内存
            del v1_arr_cpu, v2_arr_cpu, c1_arr_cpu, c2_arr_cpu

            mask_valid = None
            FORCE_CONST_MODEL = False

            # 自适应掩膜获取 (GPU)
            valid_count = 0
            for lvl in [0, 1, 2]:
                tmp_mask, desc = get_adaptive_mask_gpu(c1_arr, c2_arr, lvl)
                cnt = int(cp.sum(tmp_mask))
                if cnt >= TARGET_MIN_POINTS:
                    mask_valid = tmp_mask
                    valid_count = cnt
                    logging.info(f"{pair_tag} -> Mask Level {lvl} ({desc}). Points: {cnt}")
                    break
                else:
                    if lvl == 2:
                        mask_valid = tmp_mask
                        valid_count = cnt
                        if cnt < 500:
                            FORCE_CONST_MODEL = True

            if mask_valid is None: return None, 0
            mask_valid &= (cp.isfinite(v1_arr) & cp.isfinite(v2_arr))
            final_count = int(cp.sum(mask_valid))
            if final_count < 100:
                return None, 0

            # 3. 准备 Pareto & ALS 数据 (GPU 操作)
            # 使用 ravel 和 boolean indexing 提取有效点
            v1 = v1_arr[mask_valid].ravel().astype(cp.float64)
            v2 = v2_arr[mask_valid].ravel().astype(cp.float64)
            w1 = cp.clip(c1_arr[mask_valid].ravel(), eps_w, 1.0) ** 2
            w2 = cp.clip(c2_arr[mask_valid].ravel(), eps_w, 1.0) ** 2
            wsum = w1 + w2
            t = (w1 * v1 + w2 * v2) / (wsum + eps_w)

            # GPU 归一化
            t, b1, b2, c1, c2 = normalize_t_and_b_gpu(t, 1.0, 1.0, 0.0, 0.0, wsum=wsum)

            chosen_model = "const"
            lambda_delta_best = 0.1
            if FORCE_CONST_MODEL:
                chosen_model = "const"
            else:
                all_points = []
                for lmbd in [0.05, 0.1, 0.2]:
                    # 使用 GPU 版评估函数
                    all_points.extend(evaluate_all_models_gpu(v1, v2, t, w1, w2, wsum, lmbd))
                best_point = ideal_point_choice(pareto_frontier(all_points))
                if best_point:
                    chosen_model = best_point["model"]
                    lambda_delta_best = best_point["lambda"]

            als_offset = 0.0
            scale = 1.0

            if chosen_model == "const":
                if cp.sum(wsum) > 0:
                    als_offset = cp.asnumpy(cp.average(v1 - v2, weights=wsum))
                else:
                    als_offset = cp.asnumpy(cp.mean(v1 - v2))
            else:
                try:
                    # ALS 迭代 (GPU)
                    a1, b1_val, a2, b2_val = 0.0, 1.0, 0.0, 1.0
                    for it in range(15):
                        denom = w1 * (b1_val ** 2) + w2 * (b2_val ** 2) + eps_w
                        t_new = (w1 * b1_val * (v1 - a1) + w2 * b2_val * (v2 - a2)) / denom
                        t_new, b1_val, b2_val, _, _ = normalize_t_and_b_gpu(t_new, b1_val, b2_val, 0.0, 0.0, wsum=wsum)
                        params_lr = joint_linreg_gpu(v1, v2, t_new, w1, w2, wsum, lambda_delta_best)
                        a1_new, b1_new, a2_new, b2_new = params_lr

                        if abs(a1 - a1_new) < 1e-4: break
                        a1, b1_val, a2, b2_val = a1_new, b1_new, a2_new, b2_new
                        t = t_new

                    b2_s = b2_val if abs(b2_val) > 1e-6 else 1.0
                    scale = b1_val / b2_s
                    offset = a1 - (a2 * scale)
                    als_offset = float(offset)
                except Exception as e:
                    logging.error(f"ALS failed: {e}")
                    als_offset = cp.asnumpy(cp.nanmean(v1 - v2))

            if np.isnan(als_offset): return None, 0

            # 4. 空间拟合 (Fitting) - 使用 GPU 加速最小二乘
            v2_align = v2_arr * scale + als_offset if chosen_model != "const" else v2_arr + als_offset
            diff_map = v1_arr - v2_align
            mask_diff = cp.isfinite(diff_map) & mask_valid

            sp_params = np.zeros(6)
            diff_count = int(cp.sum(mask_diff))

            if not FORCE_CONST_MODEL and diff_count > 500:
                # 获取 mask 的索引 (GPU)
                rows_gpu, cols_gpu = cp.where(mask_diff)

                # 计算物理坐标 (GPU)
                # transform = (a, b, c, d, e, f)
                # x = a * col + b * row + c
                # y = d * col + e * row + f
                # Rasterio transform is (sx, 0, tx, 0, sy, ty) usually

                # Use cupy arrays for calculation
                # transform object is CPU, extract scalars
                ta, tb, tc = transform.a, transform.b, transform.c
                td, te, tf = transform.d, transform.e, transform.f

                px_local = ta * cols_gpu + tb * rows_gpu + tc
                py_local = td * cols_gpu + te * rows_gpu + tf

                px_local -= self.ref_x
                py_local -= self.ref_y

                z_pts = diff_map[mask_diff]
                w_fit = c1_arr[mask_diff] * c2_arr[mask_diff]
                W_sqrt = cp.sqrt(w_fit)  # vector

                try:
                    # 构造设计矩阵 A (GPU)
                    # A * x = z
                    # Weighted LS: (W_sqrt * A) * x = W_sqrt * z

                    ones_pts = cp.ones_like(z_pts)

                    if self.fitting_order == 2:
                        fit_type = "Quadratic (Order 2)"
                        # Columns: 1, x, y, x2, y2, xy
                        A = cp.column_stack((
                            ones_pts,
                            px_local,
                            py_local,
                            px_local ** 2,
                            py_local ** 2,
                            px_local * py_local
                        ))
                    else:
                        fit_type = "Linear (Order 1)"
                        A = cp.column_stack((
                            ones_pts,
                            px_local,
                            py_local
                        ))

                    # Apply weights
                    A_w = A * W_sqrt[:, None]
                    z_w = z_pts * W_sqrt

                    # GPU Solve
                    C, _, _, _ = cp.linalg.lstsq(A_w, z_w, rcond=None)

                    # Transfer result back to CPU
                    res = cp.asnumpy(C)

                    sp_params[0] = res[0]
                    sp_params[1] = res[1]
                    sp_params[2] = res[2]
                    if self.fitting_order == 2 and len(res) >= 6:
                        sp_params[3] = res[3]
                        sp_params[4] = res[4]
                        sp_params[5] = res[5]

                    logging.info(f"  -> Spatial Fit: {fit_type}. Params: {sp_params[:3]}...")

                except Exception as e:
                    logging.error(f"  -> Spatial Fit Failed: {e}")

            total_params = np.zeros(6)
            total_params[0] = als_offset + sp_params[0]
            total_params[1:] = sp_params[1:]

            # 计算权重 (GPU)
            weight = diff_count * cp.asnumpy(cp.mean(c1_arr[mask_valid] * c2_arr[mask_valid]))
            return total_params, float(weight)

    def build_mst_and_propagate(self):
        logging.info(f"Step 1: Calculating connections (Order={self.fitting_order})...")
        adj_matrix = np.zeros((self.num_images, self.num_images))
        edge_params = {}

        for i in range(self.num_images):
            for j in range(i + 1, self.num_images):
                res = self._calculate_robust_params(i, j)
                if res:
                    params, w = res
                    if w > 1e-3:
                        edge_params[(i, j)] = params
                        edge_params[(j, i)] = -params
                        adj_matrix[i, j] = w
                        adj_matrix[j, i] = w

        # Release GPU Memory heavily used in loops
        cp.get_default_memory_pool().free_all_blocks()

        logging.info("Step 2: MST Propagation...")
        mst_matrix = minimum_spanning_tree(-adj_matrix)
        mst_sym = mst_matrix + mst_matrix.T

        dists = [np.hypot(self.centers[i][0] - self.ref_x, self.centers[i][1] - self.ref_y) for i in
                 range(self.num_images)]
        anchor = np.argmin(dists)
        logging.info(f"Anchor Image: {anchor}")

        self.final_corrections[anchor] = np.zeros(6)
        order, predecessors = breadth_first_order(mst_sym, i_start=anchor, directed=False)

        for i in range(len(order)):
            curr = order[i]
            if curr == anchor: continue
            parent = predecessors[curr]
            if parent == -9999: continue

            if (curr, parent) in edge_params:
                p_rel = edge_params[(curr, parent)]
                self.final_corrections[curr] = self.final_corrections[parent] + p_rel

    def save_model(self, filepath):
        try:
            np.save(filepath, self.final_corrections)
            logging.info(f"SUCCESS: Correction model saved to {filepath}")
        except Exception as e:
            logging.error(f"FAILED to save model: {e}")

    def generate_mosaic(self, output_path_vel, output_path_coh=None):
        """
        全 GPU 镶嵌生成流程
        注意：如果结果图幅非常大（例如超过 20000x20000），可能会显存溢出。
        这里采用一次性申请全图显存的方式，适合标准 InSAR 场景。
        """
        logging.info("Step 3: Generating Mosaic on GPU...")

        # 1. 计算输出范围
        bounds = []
        res = None;
        crs = None
        for img in self.images:
            with rasterio.open(img['vel']) as src:
                bounds.append(src.bounds)
                if res is None: res = src.res; crs = src.crs
        xmin = min(b.left for b in bounds)
        xmax = max(b.right for b in bounds)
        ymin = min(b.bottom for b in bounds)
        ymax = max(b.top for b in bounds)
        width = int(np.ceil((xmax - xmin) / res[0]))
        height = int(np.ceil((ymax - ymin) / res[1]))
        transform = from_origin(xmin, ymax, res[0], res[1])

        out_prof = {'driver': 'GTiff', 'height': height, 'width': width, 'count': 1,
                    'dtype': 'float32', 'crs': crs, 'transform': transform, 'nodata': NODATA_VALUE}

        logging.info(f"Output Grid Size: {width} x {height}")

        # 2. 初始化 GPU 累加器 (可能占用较大显存)
        try:
            sum_v_gpu = cp.zeros((height, width), dtype=cp.float32)
            sum_c_gpu = cp.zeros((height, width), dtype=cp.float32)
            sum_w_gpu = cp.zeros((height, width), dtype=cp.float32)
        except cp.cuda.memory.OutOfMemoryError:
            logging.error("OOM: Output mosaic too large for GPU VRAM. Try reducing extent or using CPU fallback.")
            return

        # 3. 逐个影像处理
        for idx, img in enumerate(self.images):
            params = self.final_corrections[idx]  # CPU numpy array

            with rasterio.open(img['vel']) as src, rasterio.open(img['coh']) as src_c:
                # CPU 读取并重投影到整个网格 (这一步依然消耗内存，但为了对齐网格最简单)
                # 优化点：只重投影到局部 bounding box 然后贴进去，但代码复杂度高。
                # 这里保持原逻辑，重投影为 full frame array。
                data_cpu = reproject_to_grid(src, out_prof)
                coh_cpu = reproject_to_grid(src_c, out_prof)

                # 上传到 GPU
                data_gpu = cp.asarray(data_cpu)
                coh_gpu = cp.asarray(coh_cpu)
                del data_cpu, coh_cpu  # 释放主机内存

                mask_gpu = cp.isfinite(data_gpu) & (data_gpu != NODATA_VALUE)

                if cp.sum(mask_gpu) > 0:
                    # 获取有效点的坐标 (GPU)
                    # 生成网格坐标 (GPU) - 只针对有效点，或者生成全图 grid?
                    # 生成全图 grid 会炸显存。使用 indices 针对 mask?
                    rows, cols = cp.where(mask_gpu)

                    # Transform params
                    ta, tb, tc = transform.a, transform.b, transform.c
                    td, te, tf = transform.d, transform.e, transform.f

                    px = ta * cols + tb * rows + tc - self.ref_x
                    py = td * cols + te * rows + tf - self.ref_y

                    # 6参数校正计算 (GPU)
                    correction = (params[0] +
                                  params[1] * px +
                                  params[2] * py +
                                  params[3] * px ** 2 +
                                  params[4] * py ** 2 +
                                  params[5] * px * py)

                    correction = cp.clip(correction, -MAX_CORRECTION, MAX_CORRECTION)

                    # 应用校正
                    data_gpu[mask_gpu] -= correction.astype(cp.float32)

                    # 距离变换 (GPU)
                    # distance_transform_edt 在 cupyx 中支持 boolean mask
                    # 如果 mask 很大，EDT 很快
                    dist_gpu = distance_transform_edt(mask_gpu)
                    dist_max = cp.max(dist_gpu)
                    if dist_max > 0:
                        dist_gpu = dist_gpu / (dist_max + 1e-6)

                    # 累加
                    sum_v_gpu[mask_gpu] += data_gpu[mask_gpu] * dist_gpu[mask_gpu]
                    sum_c_gpu[mask_gpu] += cp.nan_to_num(coh_gpu[mask_gpu]) * dist_gpu[mask_gpu]
                    sum_w_gpu[mask_gpu] += dist_gpu[mask_gpu]

            # 每次循环后清理临时 GPU 内存
            del data_gpu, coh_gpu, mask_gpu
            cp.get_default_memory_pool().free_all_blocks()
            logging.info(f"Processed image {idx + 1}/{self.num_images}")

        # 4. 归一化并下载结果 (GPU -> CPU)
        final_v_gpu = cp.full((height, width), NODATA_VALUE, dtype=cp.float32)
        final_c_gpu = cp.full((height, width), 0, dtype=cp.float32)

        valid_gpu = sum_w_gpu > 0
        final_v_gpu[valid_gpu] = sum_v_gpu[valid_gpu] / sum_w_gpu[valid_gpu]
        final_c_gpu[valid_gpu] = sum_c_gpu[valid_gpu] / sum_w_gpu[valid_gpu]

        # 下载回 CPU
        final_v = cp.asnumpy(final_v_gpu)
        final_c = cp.asnumpy(final_c_gpu)

        # 清理 GPU
        del sum_v_gpu, sum_c_gpu, sum_w_gpu, final_v_gpu, final_c_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # 5. 写入磁盘
        write_with_nodata(final_v, output_path_vel, out_prof)
        if output_path_coh:
            write_with_nodata(final_c, output_path_coh, out_prof)
        logging.info(f"Saved: {output_path_vel}")


# =========================================================
# 3. 入口：生产流程
# =========================================================
if __name__ == "__main__":
    # --- 请在此处更新你的完整数据列表 ---
    target_year = 2123

    # 替换你的实际输入
    inputs = [
        {'id': 0, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_126\vel_{target_year}_113_126.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_113_126\coh_{target_year}_113_126.tif"},
        {'id': 1, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_121\vel_{target_year}_113_121.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_113_121\coh_{target_year}_113_121.tif"},
        {'id': 2, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_116\vel_{target_year}_113_116.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_113_116\coh_{target_year}_113_116.tif"},
        {'id': 3, 'vel': fr"E:\shanxi\{target_year}\{target_year}_113_111\vel_{target_year}_113_111.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_113_111\coh_{target_year}_113_111.tif"},
        {'id': 4, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_121\vel_{target_year}_11_121.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_11_121\coh_{target_year}_11_121.tif"},
        {'id': 5, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_116\vel_{target_year}_11_116.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_11_116\coh_{target_year}_11_116.tif"},
        {'id': 6, 'vel': fr"E:\shanxi\{target_year}\{target_year}_11_111\vel_{target_year}_11_111.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_11_111\coh_{target_year}_11_111.tif"},
        {'id': 7, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_127\vel_{target_year}_40_127.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_40_127\coh_{target_year}_40_127.tif"},
        {'id': 8, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_122\vel_{target_year}_40_122.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_40_122\coh_{target_year}_40_122.tif"},
        {'id': 9, 'vel': fr"E:\shanxi\{target_year}\{target_year}_40_117\vel_{target_year}_40_117.tif", 'coh': fr"E:\shanxi\{target_year}\{target_year}_40_117\coh_{target_year}_40_117.tif"},
    ]

    # 1. 自动分组
    tracks = {}
    pattern = re.compile(r'vel_\d+_(\d+)_\d+')

    for item in inputs:
        filename = os.path.basename(item['vel'])
        match = pattern.search(filename)
        path_id = match.group(1) if match else "Unknown"
        if path_id not in tracks: tracks[path_id] = []
        tracks[path_id].append(item)

    strip_results = []

    # =================================================
    # Step 1: 同轨拼接 (Linear Mode)
    # =================================================
    for path_id, images in tracks.items():
        logging.info(f"====== Processing Track {path_id} (Linear Mode) ======")
        if len(images) == 0: continue

        out_vel = os.path.join(RESULT_DIR, f"Strip_{path_id}_vel.tif")
        out_coh = os.path.join(RESULT_DIR, f"Strip_{path_id}_coh.tif")

        app = InSARMSTMosaic(images, fitting_order=1)

        if len(images) > 1:
            app.build_mst_and_propagate()
            # [关键步骤] 保存模型文件！
            model_path = os.path.join(MODEL_DIR, f"Track_{path_id}_model.npy")
            app.save_model(model_path)

        app.generate_mosaic(out_vel, out_coh)
        strip_results.append({'id': f'strip_{path_id}', 'vel': out_vel, 'coh': out_coh})

    # =================================================
    # Step 2: 异轨拼接 (Quadratic Mode)
    # =================================================
    logging.info("====== Processing Cross-Track Mosaic (Quadratic Mode) ======")
    if len(strip_results) > 1:
        app_final = InSARMSTMosaic(strip_results, fitting_order=2)
        app_final.build_mst_and_propagate()

        # [关键步骤] 保存模型文件！
        model_path_final = os.path.join(MODEL_DIR, "CrossTrack_model.npy")
        app_final.save_model(model_path_final)

        final_vel = os.path.join(RESULT_DIR, "Final_Mosaic_Vel.tif")
        final_coh = os.path.join(RESULT_DIR, "Final_Mosaic_Coh.tif")
        app_final.generate_mosaic(final_vel, final_coh)
        logging.info("ALL DONE.")
    else:
        logging.info("Only one strip result, no cross-track mosaic needed.")