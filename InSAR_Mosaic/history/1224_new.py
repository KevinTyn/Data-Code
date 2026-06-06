# -*- coding: utf-8 -*-
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import RANSACRegressor
import logging
import os
import glob
import re

# =========================
# GPU 加速库检查与回退
# =========================
try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve, percentile_filter

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    # 如果没有 GPU，定义 CPU 版本的卷积以便代码不报错
    from scipy.ndimage import convolve as scipy_convolve
    from scipy.ndimage import percentile_filter as scipy_percentile

# =========================
# 配置与常量
# =========================
NODATA_VALUE = -9999.0
eps_w = 1e-6
OUT_DIR = r"H:\\InSAR_Mosaic_Project"

# 自动创建子目录
MODEL_DIR = os.path.join(OUT_DIR, "Models")
RESULT_DIR = os.path.join(OUT_DIR, "Results")
MASK_DIR = os.path.join(OUT_DIR, "Masks")  # [新增] 用于保存 RANSAC 内点掩膜 GeoTIFF

for d in [OUT_DIR, MODEL_DIR, RESULT_DIR, MASK_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# --- 核心参数 ---
MAX_CORRECTION = 0.5
RANSAC_THRESH = 0.02

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# =========================================================
# 1. 核心数学工具库 (保持完整)
# =========================================================

def calc_rmse(residuals):
    """计算均方根误差"""
    return np.sqrt(np.mean(residuals ** 2))


def robust_weight_function(residuals, w_current, k0=1.5, k1=3.5):
    """IGG III 抗差权重函数"""
    med_abs = np.median(np.abs(residuals))
    sigma = med_abs / 0.6745
    if sigma < 1e-7: sigma = 1e-7

    norm_res = np.abs(residuals) / sigma
    factor = np.ones_like(norm_res)

    # 降权区间
    mask_down = (norm_res > k0) & (norm_res <= k1)
    factor[mask_down] = (k0 / norm_res[mask_down]) * ((k1 - norm_res[mask_down]) / (k1 - k0)) ** 2

    # 剔除区间
    mask_cut = norm_res > k1
    factor[mask_cut] = 0.0

    # 统计信息
    n_total = len(residuals)
    n_down = np.sum(mask_down)
    n_cut = np.sum(mask_cut)

    return w_current * factor, (n_down, n_cut, n_total)


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


def write_with_nodata(arr, filename, out_profile):
    arr_out = np.where(np.isnan(arr), NODATA_VALUE, arr).astype(np.float32)
    with rasterio.open(filename, "w", **out_profile) as dst:
        dst.write(arr_out, 1)


def high_coh_mask_adaptive(arr, threshold, frac, size):
    """
    自适应高相干性掩膜提取
    自动切换 GPU (CuPy) 或 CPU (NumPy/SciPy)
    """
    if HAS_GPU:
        # GPU 版本
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
        # CPU 版本 (NumPy + SciPy)
        x = np.nan_to_num(arr, nan=-9999)
        valid = (arr != -9999) & np.isfinite(arr)
        kernel = np.ones((size, size), dtype=np.float32)
        # 使用 scipy.ndimage 进行卷积
        count_total = scipy_convolve(valid.astype(np.float32), kernel, mode="constant", cval=0.0)
        count_high = scipy_convolve((x > threshold).astype(np.float32), kernel, mode="constant", cval=0.0)
        coverage = count_high / np.maximum(count_total, 1.0)
        med = scipy_percentile(x, percentile=50, size=size, mode="nearest")
        mask = (coverage >= frac) & (med >= threshold) & (x >= threshold)
        return mask


def normalize_t_and_b(t, b1, b2, c1=0.0, c2=0.0, wsum=None):
    if wsum is not None:
        t_mean = np.average(t, weights=wsum)
        t_centered = t - t_mean
        t_std = np.sqrt(np.average(t_centered ** 2, weights=wsum))
    else:
        t_mean = t.mean()
        t_centered = t - t_mean
        t_std = np.sqrt(np.mean(t_centered ** 2))
    if t_std < 1e-12: t_std = 1.0
    t_new = t_centered / t_std
    b1_new = b1 * t_std
    b2_new = b2 * t_std
    c1_new = c1 * (t_std ** 2)
    c2_new = c2 * (t_std ** 2)
    if b1_new < 0:
        t_new = -t_new;
        b1_new = -b1_new;
        b2_new = -b2_new;
        c1_new = -c1_new;
        c2_new = -c2_new
    return t_new, b1_new, b2_new, c1_new, c2_new


def joint_linreg(v1, v2, t, w1, w2, wsum, lambda_delta=1e-2):
    X = np.column_stack([np.ones_like(t), t])
    A11 = X.T @ (w1[:, None] * X);
    A22 = X.T @ (w2[:, None] * X)
    b1_vec = X.T @ (w1 * v1);
    b2_vec = X.T @ (w2 * v2)
    Axx = X.T @ (wsum[:, None] * X)
    A11 += lambda_delta * Axx;
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx;
    A21 = A12.T
    b1_vec += lambda_delta * X.T @ (wsum * v2);
    b2_vec += lambda_delta * X.T @ (wsum * v1)
    A = np.block([[A11, A12], [A21, A22]])
    b = np.concatenate([b1_vec, b2_vec])
    params = np.linalg.solve(A + 1e-8 * np.eye(4), b)
    return params


def joint_quadreg(v1, v2, t, w1, w2, wsum, lambda_delta=1e-2):
    X = np.column_stack([np.ones_like(t), t, t ** 2])
    A11 = X.T @ (w1[:, None] * X);
    A22 = X.T @ (w2[:, None] * X)
    b1_vec = X.T @ (w1 * v1);
    b2_vec = X.T @ (w2 * v2)
    Axx = X.T @ (wsum[:, None] * X)
    A11 += lambda_delta * Axx;
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx;
    A21 = A12.T
    b1_vec += lambda_delta * X.T @ (wsum * v2);
    b2_vec += lambda_delta * X.T @ (wsum * v1)
    A = np.block([[A11, A12], [A21, A22]])
    b = np.concatenate([b1_vec, b2_vec])
    A = 0.5 * (A + A.T)
    reg = 1e-8 * np.trace(A) / max(A.shape[0], 1)
    A += reg * np.eye(A.shape[0])
    params = np.linalg.solve(A, b)
    return params


def joint_constreg(v1, v2, w1, w2, wsum, lambda_delta=1e-2):
    X = np.ones((len(v1), 1))
    A11 = X.T @ (w1[:, None] * X);
    A22 = X.T @ (w2[:, None] * X)
    b1_vec = X.T @ (w1 * v1);
    b2_vec = X.T @ (w2 * v2)
    Axx = X.T @ (wsum[:, None] * X)
    A11 += lambda_delta * Axx;
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx;
    A21 = A12.T
    b1_vec += lambda_delta * X.T @ (wsum * v2);
    b2_vec += lambda_delta * X.T @ (wsum * v1)
    A = np.block([[A11, A12], [A21, A22]])
    b = np.concatenate([b1_vec, b2_vec])
    params = np.linalg.solve(A + 1e-8 * np.eye(2), b)
    return params


def wrmse(res, w):
    res_valid = res[np.isfinite(res)]
    w_valid = w[np.isfinite(res)]
    return np.sqrt(np.sum(w_valid * res_valid ** 2) / (np.sum(w_valid) + 1e-12))


def evaluate_model_shared_t(v1, v2, w1, w2, wsum, t, lambda_delta=1e-2, model="linear"):
    if model == "const":
        a1_c, a2_c = joint_constreg(v1, v2, w1, w2, wsum, lambda_delta)
        fit1 = np.full_like(v1, a1_c);
        fit2 = np.full_like(v2, a2_c)
    elif model == "linear":
        a1_l, b1_l, a2_l, b2_l = joint_linreg(v1, v2, t, w1, w2, wsum, lambda_delta)
        fit1 = a1_l + b1_l * t;
        fit2 = a2_l + b2_l * t
    elif model == "quadratic":
        a1_q, b1_q, c1_q, a2_q, b2_q, c2_q = joint_quadreg(v1, v2, t, w1, w2, wsum, lambda_delta)
        fit1 = a1_q + b1_q * t + c1_q * t ** 2;
        fit2 = a2_q + b2_q * t + c2_q * t ** 2
    else:
        raise ValueError("Unknown model")
    r1, r2 = v1 - fit1, v2 - fit2
    diff_adj = fit1 - fit2
    return {
        "model": model, "lambda": lambda_delta,
        "wrmse1": wrmse(r1, w1), "wrmse2": wrmse(r2, w2), "wrmse_diff": wrmse(diff_adj, wsum),
        "bias_diff": np.average(diff_adj, weights=wsum)
    }


def evaluate_all_models(v1, v2, t, w1, w2, wsum, lambda_delta):
    results = []
    for model in ["const", "linear", "quadratic"]:
        scores = evaluate_model_shared_t(v1, v2, w1, w2, wsum, t, lambda_delta=lambda_delta, model=model)
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
    best = None;
    best_val = np.inf
    for p in frontier:
        d, r1, r2 = p["metrics"]
        val = np.sqrt(d ** 2 + r1 ** 2 + r2 ** 2)
        if val < best_val: best_val = val; best = p
    return best


def get_adaptive_mask(a1, a2, level):
    """
    根据不同严格程度获取掩膜
    """
    if level == 0:
        m1 = high_coh_mask_adaptive(a1, 0.75, 0.65, 3)
        m2 = high_coh_mask_adaptive(a2, 0.75, 0.65, 3)
        return m1 & m2, "Strict"
    elif level == 1:
        m1 = (a1 >= 0.40);
        m2 = (a2 >= 0.40)
        return m1 & m2, "Loose"
    else:
        m1 = (a1 >= 0.10) & np.isfinite(a1)
        m2 = (a2 >= 0.10) & np.isfinite(a2)
        return m1 & m2, "Fallback"


# =========================================================
# 2. 核心类：InSARMSTMosaic
# =========================================================
class InSARMSTMosaic:
    def __init__(self, image_list, mosaic_mode='same_track'):
        """
        mosaic_mode:
            'same_track': 同轨模式 (强制使用 const 模型，防止条带弯曲)
            'cross_track': 异轨模式 (使用 linear 模型，允许平移和线性修正)
        """
        self.images = image_list
        self.num_images = len(image_list)
        self.mosaic_mode = mosaic_mode
        self.centers = {}
        self.final_corrections = np.zeros((self.num_images, 6))
        self.ref_x = 0.0
        self.ref_y = 0.0

        # --- 策略配置 (关键修改点) ---
        if self.mosaic_mode == 'same_track':
            # 同轨：高正则化，倾向于只修 Offset
            self.fitting_order = 1
            self.min_points_thresh = 400
            self.lambda_penalty = 10.0
            logging.info(">>> Mode: Same-Track (High Penalty, Prefer Const to avoid warping)")
        else:
            # 异轨：低正则化，允许修斜坡
            self.fitting_order = 1
            self.min_points_thresh = 2000
            self.lambda_penalty = 1.0
            logging.info(">>> Mode: Cross-Track (Standard Linear Fit)")

        self._init_geometry_info()

    def _init_geometry_info(self):
        xs, ys = [], []
        for idx, img in enumerate(self.images):
            with rasterio.open(img['vel']) as src:
                l, b, r, t = src.bounds
                cx, cy = (l + r) / 2, (t + b) / 2
                xs.append(cx)
                ys.append(cy)
                self.centers[idx] = (cx, cy)
        self.ref_x = np.mean(xs)
        self.ref_y = np.mean(ys)

    def _calculate_robust_params(self, idx1, idx2):
        img1, img2 = self.images[idx1], self.images[idx2]
        pair_tag = f"Pair {idx1}-{idx2}"

        # 1. 几何重叠检查
        with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2:
            xl = max(src1.bounds.left, src2.bounds.left)
            xr = min(src1.bounds.right, src2.bounds.right)
            yb = max(src1.bounds.bottom, src2.bounds.bottom)
            yt = min(src1.bounds.top, src2.bounds.top)
            if xr <= xl or yt <= yb: return None, 0

        # 2. 读取数据
        with rasterio.open(img1['vel']) as src1, rasterio.open(img2['vel']) as src2, \
                rasterio.open(img1['coh']) as c_src1, rasterio.open(img2['coh']) as c_src2:

            width = int((xr - xl) / src1.res[0])
            height = int((yt - yb) / src1.res[1])
            if width < 3 or height < 3: return None, 0

            transform = from_origin(xl, yt, src1.res[0], src1.res[1])
            prof = {'height': height, 'width': width, 'transform': transform, 'crs': src1.crs}

            v1_arr = reproject_to_grid(src1, prof)
            v2_arr = reproject_to_grid(src2, prof)
            c1_arr = np.nan_to_num(reproject_to_grid(c_src1, prof))
            c2_arr = np.nan_to_num(reproject_to_grid(c_src2, prof))

            # --- 自适应掩膜 ---
            mask_valid = None
            levels = [0, 1, 2] if self.mosaic_mode == 'cross_track' else [0, 1, 2, 3]

            for lvl in levels:
                if lvl == 3:
                    tmp_mask = (c1_arr > 0.1) & (c2_arr > 0.1)
                    desc = "Desperate (>0.1)"
                else:
                    tmp_mask, desc = get_adaptive_mask(c1_arr, c2_arr, lvl)

                cnt = int(np.sum(tmp_mask))
                if cnt >= self.min_points_thresh:
                    mask_valid = tmp_mask
                    logging.debug(f"  {pair_tag}: Accepted Mask Level {lvl} ({desc}), Points: {cnt}")
                    break

            if mask_valid is None: return None, 0
            mask_valid &= (np.isfinite(v1_arr) & np.isfinite(v2_arr))

            if np.sum(mask_valid) < self.min_points_thresh:
                return None, 0

            # 3. 准备数据
            idx_flat = np.flatnonzero(mask_valid)
            v1 = v1_arr.ravel()[idx_flat].astype(np.float64)
            v2 = v2_arr.ravel()[idx_flat].astype(np.float64)
            w1_base = np.clip(c1_arr.ravel()[idx_flat], eps_w, 1.0) ** 2
            w2_base = np.clip(c2_arr.ravel()[idx_flat], eps_w, 1.0) ** 2

            w1, w2 = w1_base.copy(), w2_base.copy()
            wsum = w1 + w2

            # --- Pareto Optimization ---
            t_initial = (w1 * v1 + w2 * v2) / (wsum + eps_w)
            t_initial, _, _, _, _ = normalize_t_and_b(t_initial, 1.0, 1.0, wsum=wsum)

            all_points = []
            lambdas = [0.05, 0.1, 0.2] if self.mosaic_mode == 'cross_track' else [0.2, 0.5, 1.0]

            for lmbd in lambdas:
                all_points.extend(evaluate_all_models(v1, v2, t_initial, w1, w2, wsum, lmbd))

            best_point = ideal_point_choice(pareto_frontier(all_points))

            chosen_model = "linear"
            best_lambda = 0.1
            if best_point:
                chosen_model = best_point["model"]
                best_lambda = best_point["lambda"] * self.lambda_penalty

            # --- [关键策略修正] 强制降级逻辑 ---
            point_count = len(v1)
            degrade_msg = ""

            # 1. 异轨：禁止二次多项式，防止长波扭曲
            if self.mosaic_mode == 'cross_track':
                if chosen_model == 'quadratic':
                    chosen_model = 'linear'
                    degrade_msg = "(Force Linear for CT)"

            # 2. 同轨：强制使用常数模型，防止窄带香蕉效应
            if self.mosaic_mode == 'same_track':
                chosen_model = 'const'
                degrade_msg = "(Force Const for Same-Track)"

            # 3. 点数过少保护
            if point_count < 800 and chosen_model != 'const':
                chosen_model = 'const'
                degrade_msg = "(Force Const due to low pts)"

            # [LOG] 模型选择日志
            logging.info(
                f"  {pair_tag} [{point_count} pts]: Selected '{chosen_model}' {degrade_msg}, lambda={best_lambda:.2f}")

            # --- ALS 迭代 ---
            igg_stats_total = [0, 0, 0]  # Down, Cut, Total

            if chosen_model == "const":
                als_offset = np.average(v1 - v2, weights=wsum)
                scale = 1.0
            else:
                t = t_initial
                t, b1, b2, _, _ = normalize_t_and_b(t, 1.0, 1.0, wsum=wsum)
                a1, a2 = 0.0, 0.0

                max_it = 15
                for it in range(max_it):
                    denom = w1 * (b1 ** 2) + w2 * (b2 ** 2) + eps_w
                    t_new = (w1 * b1 * (v1 - a1) + w2 * b2 * (v2 - a2)) / denom
                    t_new, b1, b2, _, _ = normalize_t_and_b(t_new, b1, b2, wsum=wsum)

                    a1_new, b1_new, a2_new, b2_new = joint_linreg(v1, v2, t_new, w1, w2, wsum, best_lambda)

                    fit1 = a1_new + b1_new * t_new
                    fit2 = a2_new + b2_new * t_new

                    w1, stats1 = robust_weight_function(v1 - fit1, w1_base)
                    w2, stats2 = robust_weight_function(v2 - fit2, w2_base)
                    wsum = w1 + w2

                    if it == max_it - 1 or abs(a1 - a1_new) < 1e-5:
                        igg_stats_total[0] = stats1[0] + stats2[0]
                        igg_stats_total[1] = stats1[1] + stats2[1]
                        igg_stats_total[2] = stats1[2] + stats2[2]

                    if abs(a1 - a1_new) < 1e-5 and it > 3:
                        a1, b1, a2, b2 = a1_new, b1_new, a2_new, b2_new
                        break
                    a1, b1, a2, b2 = a1_new, b1_new, a2_new, b2_new
                    t = t_new

                b2_s = b2 if abs(b2) > 1e-6 else 1.0
                scale = b1 / b2_s
                als_offset = a1 - (a2 * scale)

                logging.info(
                    f"    -> IGG Robust: Down {igg_stats_total[0]}, Cut {igg_stats_total[1]} (Total {igg_stats_total[2]})")

            # --- Spatial Fitting (RANSAC) ---
            sp_params = np.zeros(6)

            # [新增] 初始化 2D 掩膜 (用于保存 GeoTIFF)
            final_inlier_mask_2d = np.zeros(v1_arr.shape, dtype=rasterio.uint8)
            has_valid_inliers = False

            if chosen_model != "const" and point_count > self.min_points_thresh:
                v2_align = v2_arr * scale + als_offset
                diff_map = v1_arr - v2_align
                rmse_raw = calc_rmse(diff_map[mask_valid])

                filt_size = 5 if self.mosaic_mode == 'same_track' else 7
                diff_map_filt = median_filter(diff_map, size=filt_size)

                mask_diff = np.isfinite(diff_map_filt) & mask_valid
                rmse_filt = calc_rmse(diff_map_filt[mask_diff])

                if np.sum(mask_diff) > self.min_points_thresh:
                    rows, cols = np.indices(diff_map.shape)
                    px, py = transform * (cols[mask_diff], rows[mask_diff])
                    px -= self.ref_x
                    py -= self.ref_y
                    z_pts = diff_map_filt[mask_diff]

                    if self.fitting_order == 2:
                        A = np.column_stack((np.ones_like(z_pts), px, py, px ** 2, py ** 2, px * py))
                    else:
                        A = np.column_stack((np.ones_like(z_pts), px, py))

                    try:
                        thr = RANSAC_THRESH * 1.5 if self.mosaic_mode == 'same_track' else RANSAC_THRESH
                        ransac = RANSACRegressor(min_samples=0.4, residual_threshold=thr, random_state=42)
                        ransac.fit(A, z_pts)

                        inlier_ratio = np.sum(ransac.inlier_mask_) / len(z_pts)
                        if np.sum(ransac.inlier_mask_) > self.min_points_thresh // 2:
                            res = ransac.estimator_.coef_
                            res[0] = ransac.estimator_.intercept_

                            sp_params[0] = res[0];
                            sp_params[1] = res[1];
                            sp_params[2] = res[2]
                            if self.fitting_order == 2 and len(res) >= 6:
                                sp_params[3:] = res[3:6]

                            # [核心功能] 将 1D 内点映射回 2D 掩膜
                            flat_indices = np.flatnonzero(mask_diff)
                            inlier_indices = flat_indices[ransac.inlier_mask_]
                            final_inlier_mask_2d.ravel()[inlier_indices] = 1  # 标记为1
                            has_valid_inliers = True

                            inlier_mask = ransac.inlier_mask_
                            fitted = A[inlier_mask] @ res
                            rmse_final = calc_rmse(z_pts[inlier_mask] - fitted)
                            logging.info(
                                f"    -> RANSAC ({inlier_ratio:.1%}): RMSE {rmse_raw:.4f}->{rmse_filt:.4f}->{rmse_final:.4f}")
                    except Exception as e:
                        logging.warning(f"RANSAC failed: {e}")

            # 如果没跑 RANSAC (如 const)，则用 mask_valid 作为最佳掩膜
            if not has_valid_inliers:
                final_inlier_mask_2d[mask_valid] = 1

            # [核心功能] 保存内点掩膜到 GeoTIFF
            raw_weight = np.sum(mask_valid) * np.mean(c1_arr[mask_valid] * c2_arr[mask_valid])
            if raw_weight > 1e-4:
                mask_filename = os.path.join(MASK_DIR, f"Mask_{idx1}_{idx2}.tif")
                mask_prof = prof.copy()
                mask_prof.update(dtype=rasterio.uint8, count=1, nodata=0)
                try:
                    with rasterio.open(mask_filename, 'w', **mask_prof) as dst:
                        dst.write(final_inlier_mask_2d, 1)
                except Exception as e:
                    logging.warning(f"Mask save failed: {e}")

            total_params = np.zeros(6)
            total_params[0] = als_offset + sp_params[0]
            total_params[1:] = sp_params[1:]

            # [策略] 权重增强
            final_weight = raw_weight
            if self.mosaic_mode == 'same_track':
                final_weight *= 2.0

            return total_params, final_weight

    def solve_global_network(self):
        logging.info(f"Step 2: Global Network Adjustment (LSE) - Mode: {self.mosaic_mode}")
        edges = []
        weights = []
        raw_residuals = []

        for i in range(self.num_images):
            for j in range(i + 1, self.num_images):
                res = self._calculate_robust_params(i, j)
                if res:
                    params_diff, w = res
                    min_w = 1e-4 if self.mosaic_mode == 'same_track' else 1e-3
                    if w > min_w:
                        edges.append((i, j, params_diff))
                        weights.append(w)
                        raw_residuals.append(params_diff[0])

        if not edges:
            logging.warning("No valid edges found! Skipping adjustment.")
            return

        num_params = 6
        num_eq = len(edges) * num_params
        num_unk = self.num_images * num_params

        A = lil_matrix((num_eq, num_unk), dtype=np.float32)
        L = np.zeros(num_eq, dtype=np.float32)

        row_ptr = 0
        for (i, j, p_diff) in edges:
            w_edge = np.sqrt(weights[edges.index((i, j, p_diff))])
            for k in range(num_params):
                col_i = i * num_params + k
                col_j = j * num_params + k
                A[row_ptr, col_i] = 1.0 * w_edge
                A[row_ptr, col_j] = -1.0 * w_edge
                L[row_ptr] = p_diff[k] * w_edge
                row_ptr += 1

        dists = [np.hypot(self.centers[i][0] - self.ref_x, self.centers[i][1] - self.ref_y) for i in
                 range(self.num_images)]
        anchor = np.argmin(dists)

        # 添加 Anchor 约束
        for k in range(num_params):
            new_row = A.shape[0]
            A.resize((new_row + 1, num_unk))
            L = np.append(L, 0.0)
            col_anchor = anchor * num_params + k
            # 这是一个非常强的约束，固定基准影像参数为0
            A[new_row, col_anchor] = 1.0 * 1000.0

        res = lsqr(A, L, damp=1e-6)
        sol = res[0]

        # [LOG] 网络平差质量评估
        final_residual_vector = L - A @ sol
        valid_eq_count = len(edges) * num_params
        relevant_residuals = final_residual_vector[:valid_eq_count]
        sigma0 = np.sqrt(np.mean(relevant_residuals ** 2))
        avg_offset_diff = np.mean(np.abs(raw_residuals)) if raw_residuals else 0.0

        logging.info("=" * 40)
        logging.info(f"[PRECISION] Network Adjustment Result ({self.mosaic_mode}):")
        logging.info(f"  -> Total Valid Edges: {len(edges)}")
        logging.info(f"  -> Sigma0 (Unit Weight RMSE): {sigma0:.5f}")
        logging.info(f"  -> Avg Raw Offset Discrepancy: {avg_offset_diff:.4f}")
        logging.info(f"  -> Anchor Image Index: {anchor}")
        logging.info("=" * 40)

        for i in range(self.num_images):
            self.final_corrections[i] = sol[i * num_params: (i + 1) * num_params]

    def save_model(self, filepath):
        try:
            np.save(filepath, self.final_corrections)
            logging.info(f"SUCCESS: Correction model saved to {filepath}")
        except Exception as e:
            logging.error(f"FAILED to save model: {e}")

    def generate_mosaic(self, output_path_vel, output_path_coh=None):
        logging.info("Step 3: Generating Mosaic...")
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

        out_prof = {'driver': 'GTiff', 'height': height, 'width': width, 'count': 1,
                    'dtype': 'float32', 'crs': crs, 'transform': transform, 'nodata': NODATA_VALUE}

        try:
            # 内存保护：尝试分配大数组
            sum_v = np.zeros((height, width), dtype=np.float32)
            sum_c = np.zeros((height, width), dtype=np.float32)
            sum_w = np.zeros((height, width), dtype=np.float32)
        except MemoryError:
            logging.error(f"Mosaic size {width}x{height} too large for RAM! Consider block processing.")
            return

        for idx, img in enumerate(self.images):
            params = self.final_corrections[idx]
            with rasterio.open(img['vel']) as src, rasterio.open(img['coh']) as src_c:
                data = reproject_to_grid(src, out_prof)
                coh = reproject_to_grid(src_c, out_prof)
                coh = np.nan_to_num(coh)
                mask = np.isfinite(data) & (data != NODATA_VALUE)
                if np.sum(mask) > 0:
                    rows, cols = np.indices(data.shape)
                    px, py = transform * (cols[mask], rows[mask])
                    px -= self.ref_x;
                    py -= self.ref_y
                    correction = (params[0] + params[1] * px + params[2] * py +
                                  params[3] * px ** 2 + params[4] * py ** 2 + params[5] * px * py)
                    correction = np.clip(correction, -MAX_CORRECTION, MAX_CORRECTION)
                    data[mask] -= correction.astype(np.float32)

                    dist = distance_transform_edt(mask)
                    dist = dist / (np.max(dist) + 1e-6)
                    qual = coh[mask] ** 2
                    combined_w = dist[mask] * qual

                    sum_v[mask] += data[mask] * combined_w
                    sum_c[mask] += coh[mask] * combined_w
                    sum_w[mask] += combined_w

        final_v = np.full((height, width), NODATA_VALUE, dtype=np.float32)
        final_c = np.full((height, width), 0, dtype=np.float32)
        valid = sum_w > 1e-6
        final_v[valid] = sum_v[valid] / sum_w[valid]
        final_c[valid] = sum_c[valid] / sum_w[valid]
        write_with_nodata(final_v, output_path_vel, out_prof)
        if output_path_coh: write_with_nodata(final_c, output_path_coh, out_prof)
        logging.info(f"Saved: {output_path_vel}")


if __name__ == "__main__":
    # --- 输入数据 ---
    # [修复] 修正了文件路径中多余的 .tif
    target_year =2123

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

    tracks = {}
    pattern = re.compile(r'vel_\d+_(\d+)_\d+')
    for item in inputs:
        filename = os.path.basename(item['vel'])
        match = pattern.search(filename)
        path_id = match.group(1) if match else "Unknown"
        if path_id not in tracks: tracks[path_id] = []
        tracks[path_id].append(item)

    strip_results = []

    # ===============================================
    # 阶段 1：同轨拼接 (Same-Track Mosaic)
    # 重点：强制 Const 模型，防止窄带弯曲
    # ===============================================
    for path_id, images in tracks.items():
        logging.info(f"====== Processing Track {path_id} (Same-Track Mode) ======")
        if len(images) == 0: continue

        out_vel = os.path.join(RESULT_DIR, f"Strip_{path_id}_vel.tif")
        out_coh = os.path.join(RESULT_DIR, f"Strip_{path_id}_coh.tif")

        # 使用 same_track 模式，内部会自动强制 degradation 到 const
        app = InSARMSTMosaic(images, mosaic_mode='same_track')

        if len(images) > 1:
            app.solve_global_network()
            app.save_model(os.path.join(MODEL_DIR, f"Track_{path_id}_model.npy"))

        app.generate_mosaic(out_vel, out_coh)
        strip_results.append({'id': f'strip_{path_id}', 'vel': out_vel, 'coh': out_coh})

    # ===============================================
    # 阶段 2：异轨拼接 (Cross-Track Mosaic)
    # 重点：使用 Linear 模型，利用大重叠区校正
    # ===============================================
    logging.info("====== Processing Cross-Track Mosaic (Cross-Track Mode) ======")
    if len(strip_results) > 1:
        app_final = InSARMSTMosaic(strip_results, mosaic_mode='cross_track')

        app_final.solve_global_network()
        app_final.save_model(os.path.join(MODEL_DIR, "CrossTrack_model.npy"))

        final_vel = os.path.join(RESULT_DIR, "Final_Mosaic_Vel.tif")
        final_coh = os.path.join(RESULT_DIR, "Final_Mosaic_Coh.tif")
        app_final.generate_mosaic(final_vel, final_coh)
        logging.info("ALL DONE.")
    else:
        logging.info("Not enough strips for Cross-Track mosaic. Finished.")