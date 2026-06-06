# -*- coding: utf-8 -*-
"""Compute exact MAE/STD for top-9 worst pairs in yr_18 and yr_19, output as arrays."""

import sys, os, importlib.util, pandas as pd, numpy as np, time

# ── load compare_efficiency module ──
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CORE_PATH   = os.path.join(PROJECT_DIR, '1224_new2.py')
spec = importlib.util.spec_from_file_location("core", CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

from compare_efficiency import read_pair_overlap, process_one_pair
eps_w = core.eps_w

# ── extended simple functions with MAE/STD ────────
def const_metrics(v1, v2, w1, w2):
    """Return (rmse, mae, std) for simple constant adjustment."""
    wsum = w1 + w2
    diff = v1 - v2
    valid = np.isfinite(diff)
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan
    wi = wsum[valid]; dv = diff[valid]
    off = np.average(dv, weights=wi)
    res = dv - off
    rmse = np.sqrt(np.average(res**2, weights=wi))
    mae  = np.average(np.abs(res), weights=wi)
    std  = np.sqrt(np.average((res - np.average(res, weights=wi))**2, weights=wi))
    return rmse, mae, std

def linear_metrics(v1, v2, w1, w2):
    """Return (rmse, mae, std) for simple linear adjustment."""
    wsum = w1 + w2
    valid = np.isfinite(v1) & np.isfinite(v2)
    nv = np.sum(valid)
    if nv < 5:
        return np.nan, np.nan, np.nan
    t_raw = (w1 * v1 + w2 * v2) / (wsum + eps_w)
    wi = wsum[valid]; tv = t_raw[valid]; dv = (v1 - v2)[valid]
    tm = np.average(tv, weights=wi)
    ts = np.sqrt(np.average((tv - tm)**2, weights=wi))
    if ts < 1e-12:
        return np.nan, np.nan, np.nan
    tn = (tv - tm) / ts
    X = np.column_stack([np.ones_like(tn), tn])
    Wx = X * wi[:, None]
    A = X.T @ Wx
    bv = X.T @ (wi * dv)
    try:
        p = np.linalg.solve(A + 1e-8 * np.eye(2), bv)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    fit = p[0] + p[1] * tn
    res = dv - fit
    rmse = np.sqrt(np.average(res**2, weights=wi))
    mae  = np.average(np.abs(res), weights=wi)
    std  = np.sqrt(np.average((res - np.average(res, weights=wi))**2, weights=wi))
    return rmse, mae, std

# ── image list (same for all years) ─────────────
FRAMES = ['113_126','113_121','113_116','113_111','11_121','11_116','11_111','40_127','40_122','40_117']

def images_for_year(year):
    images = []
    for f in FRAMES:
        images.append({
            'vel': fr'E:\shanxi\{year}\{year}_{f}\vel_{year}_{f}.tif',
            'coh': fr'E:\shanxi\{year}\{year}_{f}\coh_{year}_{f}.tif',
        })
    for i, d in enumerate(images):
        d['id'] = i
    return images

# ── process ─────────────────────────────────────
for yr in [18, 19]:
    csv_path = os.path.join(SCRIPT_DIR, f'yr_{yr}', 'results', 'comparison_results.csv')
    df = pd.read_csv(csv_path)
    df['const_mm'] = df['c_rmse'] * 1000
    worst = df.nlargest(9, 'const_mm')
    pair_strs = worst['pair'].values

    images = images_for_year(yr)
    if not os.path.exists(images[0]['vel']):
        print(f'Year {yr}: data not found, skipping')
        continue

    labels  = []
    rmse_c_list, mae_c_list, std_c_list = [], [], []
    rmse_l_list, mae_l_list, std_l_list = [], [], []

    for ps in pair_strs:
        a, b = map(int, ps.split('-'))
        labels.append(ps)
        print(f'  {yr} pair {ps} … ', end='', flush=True)
        overlap = read_pair_overlap(images[a], images[b])
        if overlap is None:
            print('SKIP')
            continue
        v1, v2, w1, w2, nc = overlap
        rc, mc, sc = const_metrics(v1, v2, w1, w2)
        rl, ml, sl = linear_metrics(v1, v2, w1, w2)
        print(f'c=({rc*1e3:.2f},{mc*1e3:.2f},{sc*1e3:.2f}) l=({rl*1e3:.2f},{ml*1e3:.2f},{sl*1e3:.2f})')
        rmse_c_list.append(rc*1000); mae_c_list.append(mc*1000); std_c_list.append(sc*1000)
        rmse_l_list.append(rl*1000); mae_l_list.append(ml*1000); std_l_list.append(sl*1000)

    print(f'\n# ── Year {yr} ──')
    print(f'scene_labels = {labels}')
    print(f'rmse_ori = np.array([{", ".join(f"{v:.2f}" for v in rmse_c_list)}])')
    print(f'rmse_cor = np.array([{", ".join(f"{v:.2f}" for v in rmse_l_list)}])')
    print(f'mae_ori  = np.array([{", ".join(f"{v:.2f}" for v in mae_c_list)}])')
    print(f'mae_cor  = np.array([{", ".join(f"{v:.2f}" for v in mae_l_list)}])')
    print(f'std_ori  = np.array([{", ".join(f"{v:.2f}" for v in std_c_list)}])')
    print(f'std_cor  = np.array([{", ".join(f"{v:.2f}" for v in std_l_list)}])')
    print()
