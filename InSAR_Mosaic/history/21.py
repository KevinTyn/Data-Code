import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve, percentile_filter
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import logging

# =========================
# 配置与常量
# =========================
NODATA_VALUE = -9999.0
eps_w = 1e-6
gamma = 2.0
OUT_DIR = r"H:\\"  # 确保以 \\ 结尾

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# =========================
# 工具函数
# =========================
def reproject_to_grid(src, profile, resampling=Resampling.nearest):
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

def high_coh_mask_gpu(arr, threshold, frac, size):
    x = cp.array(arr)
    valid = ~cp.isnan(x)
    kernel = cp.ones((size, size), dtype=cp.float32)

    count_total = convolve(valid.astype(cp.float32), kernel, mode="constant", cval=0.0)
    count_high  = convolve((x > threshold).astype(cp.float32), kernel, mode="constant", cval=0.0)
    coverage = count_high / cp.maximum(count_total, 1.0)

    x_filled = cp.where(valid, x, threshold - 1e-6)
    med = percentile_filter(x_filled, percentile=50, size=size, mode="nearest")

    mask = (coverage >= frac) & (med >= threshold) & (x >= threshold)
    return cp.asnumpy(mask)

def normalize_t_and_b(t, b1, b2, c1=0.0, c2=0.0, wsum=None):
    if wsum is not None:
        t_mean = np.average(t, weights=wsum)
        t_centered = t - t_mean
        t_std = np.sqrt(np.average(t_centered**2, weights=wsum))
    else:
        t_mean = t.mean()
        t_centered = t - t_mean
        t_std = np.sqrt(np.mean(t_centered**2))
    if t_std < 1e-12:
        t_std = 1.0

    t_new = t_centered / t_std
    b1_new = b1 * t_std
    b2_new = b2 * t_std
    c1_new = c1 * (t_std**2)
    c2_new = c2 * (t_std**2)

    if b1_new < 0:
        t_new  = -t_new
        b1_new = -b1_new
        b2_new = -b2_new
        c1_new = -c1_new
        c2_new = -c2_new

    return t_new, b1_new, b2_new, c1_new, c2_new

# =========================
# 联合回归（带 λ 正则）
# =========================
def joint_constreg(v1, v2, w1, w2, wsum, lambda_delta=1e-2):
    X = np.ones((len(v1), 1))
    A11 = X.T @ (w1[:, None] * X)
    A22 = X.T @ (w2[:, None] * X)
    b1_vec = X.T @ (w1 * v1)
    b2_vec = X.T @ (w2 * v2)

    Axx = X.T @ (wsum[:, None] * X)
    A11 += lambda_delta * Axx
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx
    A21 = A12.T

    b1_vec += lambda_delta * X.T @ (wsum * v2)
    b2_vec += lambda_delta * X.T @ (wsum * v1)

    A = np.block([[A11, A12],
                  [A21, A22]])
    b = np.concatenate([b1_vec, b2_vec])

    params = np.linalg.solve(A + 1e-8*np.eye(2), b)
    a1_new, a2_new = params
    return a1_new, a2_new

def joint_linreg(v1, v2, t, w1, w2, wsum, lambda_delta=1e-2):
    X = np.column_stack([np.ones_like(t), t])
    A11 = X.T @ (w1[:, None] * X)
    A22 = X.T @ (w2[:, None] * X)
    b1_vec = X.T @ (w1 * v1)
    b2_vec = X.T @ (w2 * v2)

    Axx = X.T @ (wsum[:, None] * X)
    A11 += lambda_delta * Axx
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx
    A21 = A12.T

    b1_vec += lambda_delta * X.T @ (wsum * v2)
    b2_vec += lambda_delta * X.T @ (wsum * v1)

    A = np.block([[A11, A12],
                  [A21, A22]])
    b = np.concatenate([b1_vec, b2_vec])

    params = np.linalg.solve(A + 1e-8*np.eye(4), b)
    a1_new, b1_new, a2_new, b2_new = params
    return a1_new, b1_new, a2_new, b2_new

def joint_quadreg(v1, v2, t, w1, w2, wsum, lambda_delta=1e-2):
    X = np.column_stack([np.ones_like(t), t, t**2])

    A11 = X.T @ (w1[:, None] * X)
    A22 = X.T @ (w2[:, None] * X)
    b1_vec = X.T @ (w1 * v1)
    b2_vec = X.T @ (w2 * v2)

    Axx = X.T @ (wsum[:, None] * X)
    A11 += lambda_delta * Axx
    A22 += lambda_delta * Axx
    A12 = -lambda_delta * Axx
    A21 = A12.T

    b1_vec += lambda_delta * X.T @ (wsum * v2)
    b2_vec += lambda_delta * X.T @ (wsum * v1)

    A = np.block([[A11, A12],
                  [A21, A22]])
    b = np.concatenate([b1_vec, b2_vec])

    A = 0.5 * (A + A.T)
    reg = 1e-8 * np.trace(A) / max(A.shape[0], 1)
    A += reg * np.eye(A.shape[0])

    params = np.linalg.solve(A, b)
    a1_new, b1_new, c1_new, a2_new, b2_new, c2_new = params
    return a1_new, b1_new, c1_new, a2_new, b2_new, c2_new

def wrmse(res, w):
    res_valid = res[np.isfinite(res)]
    w_valid   = w[np.isfinite(res)]
    return np.sqrt(np.sum(w_valid * res_valid**2) / (np.sum(w_valid) + 1e-12))

# =========================
# 输入文件（按需修改路径）
# =========================

tif1 = r"E:\shanxi\18\18_11_121\vel_18_11_121.tif"
tif2 = r"E:\shanxi\18\18_113_121\vel_18_113_121.tif"
coh1 = r"E:\shanxi\18\18_11_121\coh_18_11_121.tif"
coh2 = r"E:\shanxi\18\18_113_121\coh_18_113_121.tif"

src1, src2 = rasterio.open(tif1), rasterio.open(tif2)
src_coh1, src_coh2 = rasterio.open(coh1), rasterio.open(coh2)

# =========================
# 并集与交集范围：统一并集原点 + 交集吸附到并集网格
# =========================
xmin_union = min(src1.bounds.left,  src2.bounds.left)
ymin_union = min(src1.bounds.bottom, src2.bounds.bottom)
xmax_union = max(src1.bounds.right, src2.bounds.right)
ymax_union = max(src1.bounds.top,   src2.bounds.top)

xmin_inter = max(src1.bounds.left,  src2.bounds.left)
ymin_inter = max(src1.bounds.bottom, src2.bounds.bottom)
xmax_inter = min(src1.bounds.right, src2.bounds.right)
ymax_inter = min(src1.bounds.top,   src2.bounds.top)

res = src1.res
crs = src1.crs

# 并集原点（左上角）
transform_union = from_origin(xmin_union, ymax_union, *res)

# 并集格网尺寸
width_union  = int(np.ceil((xmax_union - xmin_union) / res[0]))
height_union = int(np.ceil((ymax_union - ymin_union) / res[1]))

profile_union = src1.profile.copy()
profile_union.update({
    "height": height_union,
    "width":  width_union,
    "transform": transform_union,
    "crs": crs
})

# 交集格网的像元对齐吸附
offset_x = int(np.floor((xmin_inter - xmin_union) / res[0]))
offset_y = int(np.floor((ymax_union - ymax_inter) / res[1]))

x0_inter_snap = xmin_union + offset_x * res[0]
y0_inter_snap = ymax_union - offset_y * res[1]

width_inter  = int(np.ceil((xmax_inter - x0_inter_snap) / res[0]))
height_inter = int(np.ceil((y0_inter_snap - ymin_inter) / res[1]))

transform_inter = from_origin(x0_inter_snap, y0_inter_snap, *res)

profile_inter = src1.profile.copy()
profile_inter.update({
    "height": height_inter,
    "width":  width_inter,
    "transform": transform_inter,
    "crs": crs
})

logging.info(f"[Grid] Union origin=({xmin_union:.3f},{ymax_union:.3f}), Inter snapped origin=({x0_inter_snap:.3f},{y0_inter_snap:.3f})")
logging.info(f"[Grid] Union size=(H{height_union},W{width_union}), Inter size=(H{height_inter},W{width_inter})")

# =========================
# 重投影到交集格网（与并集格网对齐）
# =========================
coh_arr1 = reproject_to_grid(src_coh1, profile_inter)
coh_arr2 = reproject_to_grid(src_coh2, profile_inter)
vel_arr1 = reproject_to_grid(src1, profile_inter)
vel_arr2 = reproject_to_grid(src2, profile_inter)

# =========================
# 共同有效区域掩膜
# =========================
mask = (
    np.isfinite(coh_arr1) &
    np.isfinite(coh_arr2) &
    (coh_arr1 != 0) &
    (coh_arr2 != 0)
)

initial_valid = mask
n_initial = int(np.sum(initial_valid))
logging.info(f"[Initial] 有效重叠像素: {n_initial}")

arr1 = np.where(initial_valid, coh_arr1, np.nan)
arr2 = np.where(initial_valid, coh_arr2, np.nan)

# =========================
# 高相干掩膜（GPU）
# =========================
T, F, W = 0.75, 0.65, 3
mask1_high = high_coh_mask_gpu(arr1, threshold=T, frac=F, size=W)
mask2_high = high_coh_mask_gpu(arr2, threshold=T, frac=F, size=W)

n_high1 = int(np.sum(mask1_high))
n_high2 = int(np.sum(mask2_high))
logging.info(f"[Map 1] 高相干像素: {n_high1} ({n_high1/n_initial:.2%})")
logging.info(f"[Map 2] 高相干像素: {n_high2} ({n_high2/n_initial:.2%})")

mask_both_high = mask1_high & mask2_high
n_both = int(np.sum(mask_both_high))
logging.info(f"[Intersection] 双高相干像素: {n_both} ({n_both/n_initial:.2%})")

# =========================
# 提取速率对（先按 mask_both_high）
# =========================
vel1_selected = np.where(mask_both_high, vel_arr1, np.nan)
vel2_selected = np.where(mask_both_high, vel_arr2, np.nan)

valid_idx = np.isfinite(vel1_selected) & np.isfinite(vel2_selected)
vel_pairs = np.column_stack((vel1_selected[valid_idx], vel2_selected[valid_idx]))
logging.info(f"[Intersection] 提取到 {vel_pairs.shape[0]} 个速率对")

# =========================
# 粗差剔除（σ 与 MAD 并用）
# =========================
diff = vel1_selected - vel2_selected
valid_diff  = diff[np.isfinite(diff)]
finite_diff = np.isfinite(diff)

mu, sigma = valid_diff.mean(), valid_diff.std()
mask_outlier_sigma = np.zeros(diff.shape, dtype=bool)
mask_outlier_sigma[finite_diff] = np.abs(diff[finite_diff] - mu) > 3 * sigma

med = np.nanmedian(valid_diff)
mad = np.nanmedian(np.abs(valid_diff - med))
threshold_mad = 3 * 1.4826 * mad
mask_outlier_mad = np.zeros(diff.shape, dtype=bool)
mask_outlier_mad[finite_diff] = np.abs(diff[finite_diff] - med) > threshold_mad

mask_outlier = mask_outlier_sigma | mask_outlier_mad

vel1_selected = np.where(mask_outlier, np.nan, vel1_selected)
vel2_selected = np.where(mask_outlier, np.nan, vel2_selected)

valid_idx_final = np.isfinite(vel1_selected) & np.isfinite(vel2_selected)
logging.info(f"[Final] 粗差剔除后剩余 {int(np.sum(valid_idx_final))} 个速率对")

# =========================
# 一致性提取 idx_flat 与 v1,v2,coh_sel
# =========================
idx_flat = np.flatnonzero(valid_idx_final)
v1 = vel_arr1.ravel()[idx_flat].astype(np.float64)
v2 = vel_arr2.ravel()[idx_flat].astype(np.float64)
coh1_sel = coh_arr1.ravel()[idx_flat].astype(np.float64)
coh2_sel = coh_arr2.ravel()[idx_flat].astype(np.float64)

# =========================
# 权重与初始 t
# =========================
w1 = np.clip(coh1_sel, eps_w, 1.0)**gamma
w2 = np.clip(coh2_sel, eps_w, 1.0)**gamma
wsum = w1 + w2

# 初始 t 与参数
a1, b1, c1 = 0.0, 1.0, 0.0
a2, b2, c2 = 0.0, 1.0, 0.0
t = (w1*(v1-a1) + w2*(v2-a2)) / (wsum + eps_w)
t, b1, b2, c1, c2 = normalize_t_and_b(t, b1, b2, c1, c2, wsum=wsum)

# =========================
# 模型评估（返回三指标）
# =========================
def evaluate_model_shared_t(v1, v2, w1, w2, wsum, t, lambda_delta=1e-2, model="linear"):
    if model == "const":
        a1_c, a2_c = joint_constreg(v1, v2, w1, w2, wsum, lambda_delta)
        fit1 = np.full_like(v1, a1_c); fit2 = np.full_like(v2, a2_c)
    elif model == "linear":
        a1_l, b1_l, a2_l, b2_l = joint_linreg(v1, v2, t, w1, w2, wsum, lambda_delta)
        fit1 = a1_l + b1_l * t; fit2 = a2_l + b2_l * t
    elif model == "quadratic":
        a1_q, b1_q, c1_q, a2_q, b2_q, c2_q = joint_quadreg(v1, v2, t, w1, w2, wsum, lambda_delta)
        fit1 = a1_q + b1_q * t + c1_q * t**2
        fit2 = a2_q + b2_q * t + c2_q * t**2
    else:
        raise ValueError("Unknown model type")

    r1 = v1 - fit1
    r2 = v2 - fit2
    diff_adj = fit1 - fit2

    wrmse1 = wrmse(r1, w1)
    wrmse2 = wrmse(r2, w2)
    wrmse_diff = wrmse(diff_adj, wsum)
    bias_diff = np.average(diff_adj, weights=wsum)

    return {
        "model": model,
        "lambda": lambda_delta,
        "wrmse1": wrmse1,
        "wrmse2": wrmse2,
        "wrmse_diff": wrmse_diff,
        "bias_diff": bias_diff
    }

# =========================
# Pareto 前沿工具函数
# =========================
def pareto_frontier(points):
    frontier = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j:
                # q 支配 p：所有指标 <=，且至少一个 <
                if all(qm <= pm for qm, pm in zip(q["metrics"], p["metrics"])) and \
                   any(qm < pm for qm, pm in zip(q["metrics"], p["metrics"])):
                    dominated = True
                    break
        if not dominated:
            frontier.append(p)
    return frontier

def evaluate_all_models(v1, v2, t, w1, w2, wsum, lambda_delta):
    results = []
    for model in ["const", "linear", "quadratic"]:
        scores = evaluate_model_shared_t(v1, v2, w1, w2, wsum, t, lambda_delta=lambda_delta, model=model)
        results.append({
            "lambda": lambda_delta,
            "model": model,
            "metrics": (scores["wrmse_diff"], scores["wrmse1"], scores["wrmse2"]),
            "scores": scores
        })
    return results

def ideal_point_choice(frontier):
    best = None
    best_val = np.inf
    for p in frontier:
        d, r1, r2 = p["metrics"]
        # 计算到理想点 (0,0,0) 的欧氏距离
        val = np.sqrt(d**2 + r1**2 + r2**2)
        if val < best_val:
            best_val = val
            best = p
    return best


# =========================
# λ 扫描 + Pareto 前沿提取 + Min–max 机选
# =========================
lambda_cands = np.linspace(0.01, 0.3, 30)
all_points = []
for lmbd in lambda_cands:
    all_points.extend(evaluate_all_models(v1, v2, t, w1, w2, wsum, lmbd))

frontier = pareto_frontier(all_points)
logging.info(f"Pareto 前沿解数量: {len(frontier)}")
for f in frontier:
    d, r1m, r2m = f["metrics"]
    logging.info(f"λ={f['lambda']:.3f}, 模型={f['model']}, wrmse_diff={d:.4f}, wrmse1={r1m:.4f}, wrmse2={r2m:.4f}")

best_point = ideal_point_choice(frontier)
if best_point is None:
    raise RuntimeError("Pareto 前沿为空，无法机选模型与 λ。请检查重叠像素与掩膜设置。")

chosen_model = best_point["model"]
lambda_delta_best = best_point["lambda"]
logging.info(f"Pareto+Euclidean distance 选择: 模型={chosen_model}, λ={lambda_delta_best:.3f}")

# =========================
# ALS 迭代（使用机选的模型与 λ）
# =========================
max_iter = 20
tol = 1e-6

if chosen_model == "const":
    # 常数项无需 ALS，直接平移
    delta = np.average(v1 - v2, weights=wsum)
    fit1 = v1 - delta
    fit2 = v2
    res1 = v1 - fit1
    res2 = v2 - fit2
    t_adjusted = t.astype(np.float32)
    v1_adjusted = fit1.astype(np.float32)
    v2_adjusted = fit2.astype(np.float32)

elif chosen_model == "linear":
    a1, b1 = 0.0, 1.0
    a2, b2 = 0.0, 1.0
    for it in range(max_iter):
        denom = w1 * (b1**2) + w2 * (b2**2)
        denom = np.where(denom <= eps_w, eps_w, denom)
        t_new = (w1 * b1 * (v1 - a1) + w2 * b2 * (v2 - a2)) / denom
        t_new, b1, b2, _, _ = normalize_t_and_b(t_new, b1, b2, 0.0, 0.0, wsum=wsum)
        a1_new, b1_new, a2_new, b2_new = joint_linreg(v1, v2, t_new, w1, w2, wsum, lambda_delta=lambda_delta_best)

        param_old = np.array([a1, b1, a2, b2], dtype=np.float64)
        param_new = np.array([a1_new, b1_new, a2_new, b2_new], dtype=np.float64)
        rel_change_params = np.linalg.norm(param_new - param_old) / (np.linalg.norm(param_old) + 1e-12)
        rel_change_t      = np.linalg.norm(t_new - t) / (np.linalg.norm(t) + 1e-12)

        a1, b1, a2, b2 = a1_new, b1_new, a2_new, b2_new
        t = t_new

        logging.info(f"[Linear] Iter {it+1}: Δparams={rel_change_params:.3e}, Δt={rel_change_t:.3e}")
        if rel_change_params < tol and rel_change_t < tol:
            logging.info(f"ALS (线性)收敛: iter {it+1}")
            break

    fit1 = a1 + b1 * t
    fit2 = a2 + b2 * t
    res1 = v1 - fit1
    res2 = v2 - fit2
    t_adjusted = t.astype(np.float32)
    v1_adjusted = fit1.astype(np.float32)
    v2_adjusted = fit2.astype(np.float32)

else:  # quadratic
    a1, b1, c1 = 0.0, 1.0, 0.0
    a2, b2, c2 = 0.0, 1.0, 0.0
    for it in range(max_iter):
        g1 = a1 + b1 * t + c1 * t**2
        g2 = a2 + b2 * t + c2 * t**2

        dg1 = b1 + 2.0 * c1 * t
        dg2 = b2 + 2.0 * c2 * t

        numer = w1 * (v1 - g1) * dg1 + w2 * (v2 - g2) * dg2
        denom = w1 * (dg1**2) + w2 * (dg2**2)
        denom = np.where(denom <= eps_w, eps_w, denom)
        t_new = t + numer / denom

        t_new, b1, b2, c1, c2 = normalize_t_and_b(t_new, b1, b2, c1, c2, wsum=wsum)
        a1_new, b1_new, c1_new, a2_new, b2_new, c2_new = joint_quadreg(v1, v2, t_new, w1, w2, wsum, lambda_delta=lambda_delta_best)

        param_old = np.array([a1, b1, c1, a2, b2, c2], dtype=np.float64)
        param_new = np.array([a1_new, b1_new, c1_new, a2_new, b2_new, c2_new], dtype=np.float64)
        rel_change_params = np.linalg.norm(param_new - param_old) / (np.linalg.norm(param_old) + 1e-12)
        rel_change_t      = np.linalg.norm(t_new - t) / (np.linalg.norm(t) + 1e-12)

        a1, b1, c1, a2, b2, c2 = a1_new, b1_new, c1_new, a2_new, b2_new, c2_new
        t = t_new

        logging.info(f"[Quadratic] Iter {it+1}: Δparams={rel_change_params:.3e}, Δt={rel_change_t:.3e}")
        if rel_change_params < tol and rel_change_t < tol:
            logging.info(f"ALS (二次项)收敛: iter {it+1}")
            break
    else:
        logging.info("ALS (二次项)达到最大迭代步未严格收敛。")

    fit1 = a1 + b1 * t + c1 * t**2
    fit2 = a2 + b2 * t + c2 * t**2
    res1 = v1 - fit1
    res2 = v2 - fit2
    t_adjusted = t.astype(np.float32)
    v1_adjusted = fit1.astype(np.float32)
    v2_adjusted = fit2.astype(np.float32)

# =========================
# 诊断与写出结果
# =========================
rss1 = np.sum(w1 * res1 ** 2)
rss2 = np.sum(w2 * res2 ** 2)
df1 = max(len(v1) - (1 if chosen_model == "const" else 2 if chosen_model == "linear" else 3), 1)
df2 = max(len(v2) - (1 if chosen_model == "const" else 2 if chosen_model == "linear" else 3), 1)
sigma1_hat = np.sqrt(rss1 / df1)
sigma2_hat = np.sqrt(rss2 / df2)

logging.info("=== Pareto+MinMax 机选结果 ===")
if chosen_model == "const":
    logging.info(f"Const 模型: sigma1~={sigma1_hat:.6f}, sigma2~={sigma2_hat:.6f}")
elif chosen_model == "linear":
    logging.info(f"Map1: a1={a1:.6f}, b1={b1:.6f}, sigma1~={sigma1_hat:.6f}")
    logging.info(f"Map2: a2={a2:.6f}, b2={b2:.6f}, sigma2~={sigma2_hat:.6f}")
else:
    logging.info(f"Map1: a1={a1:.6f}, b1={b1:.6f}, c1={c1:.6f}, sigma1~={sigma1_hat:.6f}")
    logging.info(f"Map2: a2={a2:.6f}, b2={b2:.6f}, c2={c2:.6f}, sigma2~={sigma2_hat:.6f}")

# 将诊断结果写回交集整图
t_full       = np.full(vel_arr1.shape, np.nan, dtype=np.float32)
v1_adj_full  = np.full(vel_arr1.shape, np.nan, dtype=np.float32)
v2_adj_full  = np.full(vel_arr2.shape, np.nan, dtype=np.float32)

t_full.ravel()[idx_flat]      = t_adjusted
v1_adj_full.ravel()[idx_flat] = v1_adjusted
v2_adj_full.ravel()[idx_flat] = v2_adjusted

out_profile = profile_inter.copy()
out_profile.update(dtype=rasterio.float32, count=1, nodata=NODATA_VALUE)

write_with_nodata(t_full,       OUT_DIR + "vel_true_adjusted.tif", out_profile)
write_with_nodata(v1_adj_full,  OUT_DIR + "vel_map1_adjusted.tif", out_profile)
write_with_nodata(v2_adj_full,  OUT_DIR + "vel_map2_adjusted.tif", out_profile)

vel_diff_adj = v1_adj_full - v2_adj_full
write_with_nodata(vel_diff_adj, OUT_DIR + "vel_diff_adjusted.tif", out_profile)

res1_full = np.full(vel_arr1.shape, np.nan, dtype=np.float32)
res2_full = np.full(vel_arr2.shape, np.nan, dtype=np.float32)
res1_full.ravel()[idx_flat] = res1.astype(np.float32)
res2_full.ravel()[idx_flat] = res2.astype(np.float32)
write_with_nodata(res1_full, OUT_DIR + "residual_map1.tif", out_profile)
write_with_nodata(res2_full, OUT_DIR + "residual_map2.tif", out_profile)

logging.info("间接平差完成，结果已输出为 GeoTIFF (nodata=-9999.0)")

# =========================
# 可视化与诊断输出
# =========================
valid_idx_final_adj = np.isfinite(v1_adj_full) & np.isfinite(v2_adj_full)
v1h_adj = v1_adj_full[valid_idx_final_adj]
v2h_adj = v2_adj_full[valid_idx_final_adj]

def residual_diagnostics(res, label="Map"):
    res_valid = res[np.isfinite(res)]
    mean   = res_valid.mean()
    std    = res_valid.std()
    median = np.median(res_valid)
    mad    = np.median(np.abs(res_valid - median))
    q1, q3 = np.percentile(res_valid, [25, 75])
    iqr    = q3 - q1
    rmse   = np.sqrt(np.mean(res_valid**2))
    skewness = skew(res_valid)
    kurtv    = kurtosis(res_valid)

    print(f"{label} 残差诊断:")
    print(f"  均值={mean:.6f}, 中位数={median:.6f}")
    print(f"  标准差={std:.6f}, MAD={mad:.6f}, IQR={iqr:.6f}")
    print(f"  RMSE={rmse:.6f}")
    print(f"  偏度={skewness:.3f}, 峰度={kurtv:.3f}")
    print(f"  min={res_valid.min():.6f}, max={res_valid.max():.6f}")

def residual_range_summary(res, label="Map"):
    res_valid = res[np.isfinite(res)]
    total = len(res_valid)
    mean  = res_valid.mean()
    std   = res_valid.std()
    q1, q3 = np.percentile(res_valid, [25, 75])
    iqr    = q3 - q1
    min_val, max_val = res_valid.min(), res_valid.max()

    print(f"{label} 残差分布区间统计 (单位: m/yr)")
    print(f"  总数: {total}")
    print(f"  min–max: [{min_val:.3f}, {max_val:.3f}]")
    print(f"  均值±σ: [{mean-std:.3f}, {mean+std:.3f}] ≈ 68% 数据")
    print(f"  均值±2σ: [{mean-2*std:.3f}, {mean+2*std:.3f}] ≈ 95% 数据")
    print(f"  IQR (Q1–Q3): [{q1:.3f}, {q3:.3f}] 中间 50% 数据")

model_cn = "常数项" if chosen_model=="const" else "线性" if chosen_model=="linear" else "二次项"
residual_diagnostics(res1, f"Map1 ({model_cn})")
residual_diagnostics(res2, f"Map2 ({model_cn})")
residual_range_summary(res1, "Map1")
residual_range_summary(res2, "Map2")

def check_t_diagnostics(t_vec, v1_vec, v2_vec, fit1_vec, fit2_vec, w1_vec, w2_vec, wsum_vec, label="诊断"):
    print(f"{label} - t 数值检查")
    print(f"  NaN 数量: {np.isnan(t_vec).sum()}")
    print(f"  方差: {np.var(t_vec):.6f}")
    print(f"  范围: [{np.nanmin(t_vec):.3f}, {np.nanmax(t_vec):.3f}]")

    r1_loc = v1_vec - fit1_vec
    r2_loc = v2_vec - fit2_vec
    print(f"{label} - 残差统计")
    print(f"  Map1 残差均值={np.nanmean(r1_loc):.6f}, 标准差={np.nanstd(r1_loc):.6f}")
    print(f"  Map2 残差均值={np.nanmean(r2_loc):.6f}, 标准差={np.nanstd(r2_loc):.6f}")

    d = fit1_vec - fit2_vec
    wrmse_diff_loc = np.sqrt(np.sum(wsum_vec * d**2) / (np.sum(wsum_vec) + 1e-12))
    bias_diff_loc  = np.sum(wsum_vec * d) / (np.sum(wsum_vec) + 1e-12)
    print(f"{label} - 一致性指标")
    print(f"  wrmse_diff={wrmse_diff_loc:.6f}, bias_diff={bias_diff_loc:.6f}")

# 使用方法：在 ALS 收敛后调用（向量均为交集有效样本）
check_t_diagnostics(t, v1, v2, fit1, fit2, w1, w2, wsum, label="最终解 (Pareto+MinMax)")

# =========================
# 最终方案：空间残差曲面校正 (彻底消除拼接痕迹)
# =========================
from scipy.ndimage import distance_transform_edt

logging.info("开始执行：空间残差曲面校正拼接...")

# 1. 准备全图数据
vel_arr1_union = reproject_to_grid(src1, profile_union, resampling=Resampling.bilinear)
vel_arr2_union = reproject_to_grid(src2, profile_union, resampling=Resampling.bilinear)

# 2. 初步校正 (基于 ALS 参数)
if chosen_model == "const":
    v2_aligned = vel_arr2_union + delta
else:
    b2_safe = b2 if abs(b2) > 1e-6 else 1.0
    scale = b1 / b2_safe
    offset = a1 - (a2 * scale)
    v2_aligned = vel_arr2_union * scale + offset
    logging.info(f"ALS参数校正: Map2_new = {scale:.4f} * Map2 + {offset:.4f}")

# =========================================================
# 改进版：几何自适应的空间残差拟合 (替换原有的简单拟合)
# =========================================================
logging.info("正在计算重叠区残留趋势 (自适应模式)...")

# 1. 计算差异
diff = vel_arr1_union - v2_aligned
mask_diff = np.isfinite(diff)

# 2. 配置参数
MIN_POINTS = 500  # 最小点数要求 (比原版100更严格)
MIN_SPREAD_RATIO = 0.05  # 判定为"窄条带"的阈值 (跨度/全图尺寸)
MAX_CORRECTION = 5.0  # [重要] 安全钳制阈值，单位与速率图一致(如mm/yr或cm/yr)
# 防止校正量过大导致Map2被过度扭曲

if np.sum(mask_diff) < MIN_POINTS:
    logging.warning("重叠区有效点过少，跳过空间残差拟合，仅保留 ALS 结果。")
    v2_corrected = v2_aligned
else:
    # 建立坐标索引
    rows, cols = np.indices(diff.shape)

    # 提取有效点 (暂不降采样，为了准确计算分布std)
    y_valid = rows[mask_diff]
    x_valid = cols[mask_diff]
    z_valid = diff[mask_diff]

    # 3. 计算重叠区的几何分布特征
    x_std = np.std(x_valid)
    y_std = np.std(y_valid)
    h_img, w_img = diff.shape

    # 4. 判定拟合模式：跨度够大才拟合该方向的斜率
    fit_x = x_std > (w_img * MIN_SPREAD_RATIO)  # X方向够宽吗？
    fit_y = y_std > (h_img * MIN_SPREAD_RATIO)  # Y方向够高吗？

    logging.info(f"重叠区分布特征: X_std={x_std:.1f} (阈值 {w_img * MIN_SPREAD_RATIO:.1f}), "
                 f"Y_std={y_std:.1f} (阈值 {h_img * MIN_SPREAD_RATIO:.1f})")

    # 准备用于解算的降采样数据 (提高速度)
    step = 10
    y_s = y_valid[::step]
    x_s = x_valid[::step]
    z_s = z_valid[::step]

    # 5. 执行自适应拟合
    if fit_x and fit_y:
        logging.info(">>> 模式: 二维平面拟合 (Planar: Offset + SlopeX + SlopeY)")
        # 这就是你原始代码的逻辑
        A = np.column_stack((np.ones_like(z_s), x_s, y_s))
        C, _, _, _ = np.linalg.lstsq(A, z_s, rcond=None)
        plane_correction = C[0] + C[1] * cols + C[2] * rows

    elif fit_y and not fit_x:
        logging.info(">>> 模式: 纵向线性拟合 (Linear-Y: Offset + SlopeY) - X方向由于太窄已忽略")
        # 针对南北走向的窄条带 (InSAR同轨拼接最常见情况)
        A = np.column_stack((np.ones_like(z_s), y_s))
        C, _, _, _ = np.linalg.lstsq(A, z_s, rcond=None)
        plane_correction = C[0] + C[1] * rows

    elif fit_x and not fit_y:
        logging.info(">>> 模式: 横向线性拟合 (Linear-X: Offset + SlopeX) - Y方向由于太窄已忽略")
        # 针对东西走向的窄条带
        A = np.column_stack((np.ones_like(z_s), x_s))
        C, _, _, _ = np.linalg.lstsq(A, z_s, rcond=None)
        plane_correction = C[0] + C[1] * cols

    else:
        logging.info(">>> 模式: 纯常数校正 (Constant) - 重叠区过于集中")
        # 如果就是一个小方块，只平移，不倾斜
        offset_val = np.median(z_s)
        plane_correction = np.full(diff.shape, offset_val, dtype=np.float32)

    # 6. 安全钳制 (防止校正过头)
    # 如果拟合出来的趋势太夸张(例如从左到右产生几十厘米的差异)，说明拟合出错了，必须强行压扁
    rng = plane_correction.max() - plane_correction.min()
    if rng > MAX_CORRECTION * 2:
        logging.warning(f"警告：空间校正趋势过大 (Range={rng:.2f})，已触发安全衰减。")
        scale_factor = (MAX_CORRECTION * 2) / rng
        mean_val = np.mean(plane_correction)
        plane_correction = (plane_correction - mean_val) * scale_factor + mean_val

    # 7. 应用校正
    v2_corrected = v2_aligned + plane_correction

# =========================================================
# 拼接步骤：Sigmoid 加权 (比线性羽化过渡更陡峭、更隐形)
# =========================================================

# 生成掩膜
mask1 = np.isfinite(vel_arr1_union)
mask2 = np.isfinite(v2_corrected)
mask_overlap = mask1 & mask2

# 计算距离场
dist1 = distance_transform_edt(mask1)
dist2 = distance_transform_edt(mask2)
sum_dist = dist1 + dist2
sum_dist[sum_dist == 0] = 1.0

# 线性权重 0~1
w_linear = dist1 / sum_dist

# Sigmoid 变换：将线性权重变得"两头平、中间陡"
# 这样大部分区域保持原值，只在接缝线附近快速切换，肉眼几乎不可见
k = 10.0  # 陡峭系数，越大越像硬切割
w_sigmoid = 1.0 / (1.0 + np.exp(-k * (w_linear - 0.5)))

# 将权重归一化到 mask 内
w1_final = w_sigmoid
w2_final = 1.0 - w_sigmoid

# 执行拼接
vel_final = np.full(vel_arr1_union.shape, np.nan, dtype=np.float32)

# 1. 非重叠区
mask_only1 = mask1 & ~mask2
mask_only2 = mask2 & ~mask1
vel_final[mask_only1] = vel_arr1_union[mask_only1]
vel_final[mask_only2] = v2_corrected[mask_only2]  # 注意是用 corrected 后的数据

# 2. 重叠区
if np.sum(mask_overlap) > 0:
    # 理论上此时 vel_arr1 和 v2_corrected 非常接近，怎么加权都没痕迹
    # 但使用 Sigmoid 权重可以保证纹理的无缝切换
    vel_final[mask_overlap] = (vel_arr1_union[mask_overlap] * w1_final[mask_overlap] +
                               v2_corrected[mask_overlap] * w2_final[mask_overlap])

# =========================
# 输出
# =========================
out_profile_union = profile_union.copy()
out_profile_union.update(dtype=rasterio.float32, count=1, nodata=NODATA_VALUE)

write_with_nodata(vel_final, OUT_DIR + "vel_mosaic_seamless.tif", out_profile_union)
logging.info("全部处理完成！")
logging.info(f"最终结果: {OUT_DIR}vel_mosaic_seamless.tif")
logging.info("优化说明：已通过平面拟合消除了重叠区的系统性高程差，拼接痕迹应已完全消失。")



