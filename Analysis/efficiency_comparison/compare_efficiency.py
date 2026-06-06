# -*- coding: utf-8 -*-
"""
Efficiency Comparison: AINA System vs. Simple Constant/Linear Adjustment

Compares three approaches for InSAR image pair adjustment:
  1. Simple Constant   – weighted average offset only
  2. Simple Linear     – weighted linear regression (common-signal predictor)
  3. AINA (Full) – Pareto front + ALS iteration + robust IGG-III weights

Data: auto-generated synthetic InSAR-style pairs (no external files needed).
Output (under efficiency_comparison/):
  figures/   – 4 comparison charts (PNG, 300 DPI)
  results/   – CSV + TXT summary
"""

import sys, os, time, importlib.util, logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling

# ── Paths ──────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CORE_PATH   = os.path.join(PROJECT_DIR, '1224_new2.py')

# ── Import 1224_new2.py via importlib (filename starts with digit) ──
spec = importlib.util.spec_from_file_location("mosaic_core", CORE_PATH)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

eps_w = core.eps_w

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  Synthetic data generator
# ══════════════════════════════════════════════════════════════

def make_synthetic_pair(seed, n_points=3000):
    """
    Generate synthetic 1-d velocity & weight arrays that mimic a real InSAR overlap.
    Returns v1, v2, w1, w2 (all float64, same length).
    """
    rng = np.random.default_rng(seed)
    n = n_points + rng.integers(-500, 500)

    # --- common deformation signal ---
    t_common = rng.normal(0, 0.02, n)

    # --- systematic difference between images ---
    # constant offset + mild ramp on common signal + noise
    offset    = rng.uniform(-0.04, 0.04)       # m/yr
    ramp      = rng.uniform(-0.6, 0.6)          # coefficient multiplying common signal
    noise_std = rng.uniform(0.003, 0.010)

    v1 = t_common + rng.normal(0, noise_std, n)
    v2 = t_common + offset + ramp * t_common + rng.normal(0, noise_std, n)

    # --- weights (simulate coherence squared) ---
    w1 = np.clip(rng.beta(3, 1.2, n).astype(np.float64), 0.02, 1.0) ** 2
    w2 = np.clip(rng.beta(2.8, 1.0, n).astype(np.float64), 0.02, 1.0) ** 2

    return (v1.astype(np.float64), v2.astype(np.float64),
            w1.astype(np.float64), w2.astype(np.float64))


# ══════════════════════════════════════════════════════════════
#  Simple adjustment (no Pareto / no ALS)
# ══════════════════════════════════════════════════════════════

def simple_const_pair(v1, v2, w1, w2):
    """Weighted constant offset.  Returns (offset, rmse)."""
    wsum = w1 + w2
    diff = v1 - v2
    valid = np.isfinite(diff)
    if np.sum(valid) < 3:
        return 0.0, np.nan
    off = np.average(diff[valid], weights=wsum[valid])
    rmse = np.sqrt(np.average((diff[valid] - off) ** 2, weights=wsum[valid]))
    return off, rmse


def simple_linear_pair(v1, v2, w1, w2):
    """Weighted linear reg. on common signal t = (w1*v1 + w2*v2) / (w1+w2).
       Returns ((a, b), rmse) where  v1 - v2 ≈ a + b * t_raw."""
    wsum = w1 + w2
    valid = np.isfinite(v1) & np.isfinite(v2)
    nv = np.sum(valid)
    if nv < 5:
        return (0.0, 0.0), np.nan

    t_raw = (w1 * v1 + w2 * v2) / (wsum + eps_w)

    wi  = wsum[valid];  tv = t_raw[valid];  dv = (v1 - v2)[valid]
    tm  = np.average(tv, weights=wi)
    ts  = np.sqrt(np.average((tv - tm) ** 2, weights=wi))
    if ts < 1e-12:
        return (np.average(dv, weights=wi), 0.0), np.nan
    tn = (tv - tm) / ts

    X  = np.column_stack([np.ones_like(tn), tn])
    Wx = X * wi[:, None]
    A  = X.T @ Wx
    bv = X.T @ (wi * dv)
    try:
        p = np.linalg.solve(A + 1e-8 * np.eye(2), bv)
    except np.linalg.LinAlgError:
        a0 = np.average(dv, weights=wi)
        return (a0, 0.0), np.nan

    b_orig = p[1] / ts
    a_orig = p[0] - b_orig * tm
    resid  = dv - (p[0] + p[1] * tn)
    rmse   = np.sqrt(np.average(resid ** 2, weights=wi))
    return (a_orig, b_orig), rmse


# ══════════════════════════════════════════════════════════════
#  One pair – all three methods
# ══════════════════════════════════════════════════════════════

def process_one_pair(v1, v2, w1, w2, idx):
    wsum = w1 + w2
    n    = len(v1)
    rec  = {'pair': f'P{idx:02d}', 'n_points': n}

    # ── Method 1  Simple Constant ──────────────────────────
    t0 = time.perf_counter()
    _, rmse_c = simple_const_pair(v1, v2, w1, w2)
    rec['c_time_s'] = time.perf_counter() - t0
    rec['c_rmse']   = rmse_c

    # ── Method 2  Simple Linear ────────────────────────────
    t0 = time.perf_counter()
    _, rmse_l = simple_linear_pair(v1, v2, w1, w2)
    rec['l_time_s'] = time.perf_counter() - t0
    rec['l_rmse']   = rmse_l

    # ── Method 3  AINA (Full system) ─────────────────
    t0 = time.perf_counter()

    t_init = (w1 * v1 + w2 * v2) / (wsum + eps_w)
    t_init, _, _, _, _ = core.normalize_t_and_b(t_init, 1.0, 1.0, wsum=wsum)

    all_pts = []
    for lb in [0.2, 0.5, 1.0]:
        all_pts.extend(core.evaluate_all_models(v1, v2, t_init, w1, w2, wsum, lb))
    frontier = core.pareto_frontier(all_pts)
    best_pt  = core.ideal_point_choice(frontier)
    chosen   = best_pt['model'] if best_pt else 'const'
    best_lbd = best_pt['lambda'] if best_pt else 0.1
    if chosen == 'quadratic':
        chosen = 'const'

    # audit (same-track style, conservative)
    rmse_const_f = np.sqrt(np.average(
        (v1 - v2 - np.average(v1 - v2, weights=wsum)) ** 2, weights=wsum))
    if chosen == 'linear':
        rmse_lin_f  = best_pt['metrics'][0] if best_pt else rmse_const_f
        improvement = (rmse_const_f - rmse_lin_f) / (rmse_const_f + 1e-9)
        if (rmse_const_f < core.RMSE_TOLERANCE
            or improvement <= core.IMPROVEMENT_THRESH
            or n <= 600):
            chosen = 'const'

    # ALS iteration (IGG-III reweighting)
    if chosen == 'const':
        full_rmse = rmse_const_f
    else:
        t_c, b1, b2, _, _ = core.normalize_t_and_b(t_init, 1.0, 1.0, wsum=wsum)
        a1, a2 = 0.0, 0.0
        w1_i, w2_i = w1.copy(), w2.copy()
        for it in range(15):
            denom = w1_i * b1**2 + w2_i * b2**2 + eps_w
            t_new = (w1_i * b1 * (v1 - a1) + w2_i * b2 * (v2 - a2)) / denom
            t_new, b1, b2, _, _ = core.normalize_t_and_b(
                t_new, b1, b2, wsum=w1_i + w2_i)
            a1_n, b1_n, a2_n, b2_n = core.joint_linreg(
                v1, v2, t_new, w1_i, w2_i, w1_i + w2_i, best_lbd * 10.0)
            fit1 = a1_n + b1_n * t_new
            fit2 = a2_n + b2_n * t_new
            w1_i, _ = core.robust_weight_function(v1 - fit1, w1)
            w2_i, _ = core.robust_weight_function(v2 - fit2, w2)
            if abs(a1 - a1_n) < 1e-5 and it > 3:
                a1, b1, a2, b2 = a1_n, b1_n, a2_n, b2_n
                break
            a1, b1, a2, b2 = a1_n, b1_n, a2_n, b2_n
            t_c = t_new
        full_rmse = core.wrmse((a1 + b1 * t_c) - (a2 + b2 * t_c), w1 + w2)

    rec['f_time_s'] = time.perf_counter() - t0
    rec['f_rmse']   = full_rmse
    rec['f_model']  = chosen
    return rec


# ══════════════════════════════════════════════════════════════
#  Figure generation
# ══════════════════════════════════════════════════════════════

def make_figures(df, fig_dir):
    """Produce publication-quality comparison figures."""
    C_C  = '#2E86AB';   C_L  = '#A23B72';   C_F  = '#F18F01'
    C_IO = '#90A4AE'
    n    = len(df)
    x    = np.arange(n)
    w    = 0.25

    const_mm = df['c_rmse'].values * 1000
    lin_mm   = df['l_rmse'].values * 1000
    full_mm  = df['f_rmse'].values * 1000

    read_s  = df['read_sec'].values if 'read_sec' in df.columns else np.zeros(n)
    comp_c   = df['c_time_s'].values
    comp_l   = df['l_time_s'].values
    comp_f   = df['f_time_s'].values
    total_c  = read_s + comp_c
    total_l  = read_s + comp_l
    total_f  = read_s + comp_f

    imp_l = (const_mm - lin_mm)  / (const_mm + 1e-9) * 100
    imp_f = (const_mm - full_mm) / (const_mm + 1e-9) * 100

    n_const  = int(np.sum(df['f_model'] == 'const'))
    n_linear = int(np.sum(df['f_model'] == 'linear'))

    label_pairs = df['pair'].values

    # ── global font / style ─────────────────────────────────
    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'font.size':         9,
        'axes.titlesize':   12,
        'axes.labelsize':   10,
        'xtick.labelsize':  8,
        'ytick.labelsize':  8,
        'legend.fontsize':  8,
        'figure.dpi':       150,
    })

    # ══════════════════════════════════════════════════════════
    #  FIGURE 1 – Processing Time per Pair  (stacked: I/O + compute)
    # ══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(28, 10))

    w_stack = 0.22
    for i in range(n):
        ax.bar(i - w_stack, comp_c[i],  w_stack, color=C_C,  edgecolor='white', lw=.3, zorder=3)
        ax.bar(i - w_stack, read_s[i],   w_stack, bottom=comp_c[i], color=C_IO,
               edgecolor='white', lw=.3, zorder=3, alpha=.55)
        ax.bar(i,           comp_l[i],  w_stack, color=C_L,  edgecolor='white', lw=.3, zorder=3)
        ax.bar(i,           read_s[i],   w_stack, bottom=comp_l[i], color=C_IO,
               edgecolor='white', lw=.3, zorder=3, alpha=.55)
        ax.bar(i + w_stack, comp_f[i],  w_stack, color=C_F,  edgecolor='white', lw=.3, zorder=3)
        ax.bar(i + w_stack, read_s[i],   w_stack, bottom=comp_f[i], color=C_IO,
               edgecolor='white', lw=.3, zorder=3, alpha=.55)
        tval = total_f[i]
        ax.text(i + w_stack, tval + max(total_f)*.015,
                f'{tval:.1f}s', ha='center', va='bottom', fontsize=26, fontweight='bold',
                color='#444444')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_F,   edgecolor='white', label='AINA'),
        Patch(facecolor=C_L,   edgecolor='white', label='Linear'),
        Patch(facecolor=C_C,   edgecolor='white', label='Constant'),
        Patch(facecolor=C_IO,  edgecolor='white', label='I/O'),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right', ncol=1,
                     framealpha=.92, edgecolor='#BBBBBB',
                     prop={'family': 'serif', 'size': 28})

    ax.set_xlabel('', fontsize=34, fontweight='bold')
    ax.set_ylabel('Processing Time (s)', fontsize=34, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_pairs, rotation=0, ha='center', fontsize=22)
    ax.tick_params(axis='both', labelsize=30)
    ax.yaxis.grid(True, alpha=.18, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, n - 0.4)

    # frame index legend below x-axis
    if 'pair_frames' in df.columns:
        frame_map = df['pair_frames'].iloc[0] if len(df) > 0 else ''
        # build a compact legend of index→frame mapping
        frame_map_text = (
            '0: 113_126,  1: 113_121,  2: 113_116,  3: 113_111,  4: 11_121,  '
            '5: 11_116,  6: 11_111,  7: 40_127,  8: 40_122,  9: 40_117'
        )
    else:
        frame_map_text = ''
    ax.text(0.5, -0.10, frame_map_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=30, family='serif', color='#555555')

    # summary annotation
    sum_text = (
        f"Total: Const {np.sum(total_c):.0f}s  |  Linear {np.sum(total_l):.0f}s  |  "
        f"AINA {np.sum(total_f):.0f}s\n"
        f"Mean per pair: Const {np.mean(total_c):.1f}s  |  Linear {np.mean(total_l):.1f}s  |  "
        f"AINA {np.mean(total_f):.1f}s"
    )
    ax.text(0.02, 0.97, sum_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=28, family='serif',
            bbox=dict(boxstyle='round,pad=.5', facecolor='white', edgecolor='#BBBBBB', alpha=.9))

    # ensure bar-top numbers fit within axes
    ymax = max(total_f) * 1.15
    ax.set_ylim(0, ymax)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    p = os.path.join(fig_dir, 'processing_time.png')
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    log.info("Saved %s", p);  plt.close(fig)

    # ══════════════════════════════════════════════════════════
    #  FIGURE 2 – Average Time Breakdown  (stacked bar, academic)
    # ══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    methods     = ['Constant', 'Linear', 'AINA']
    comp_means  = [np.mean(comp_c), np.mean(comp_l), np.mean(comp_f)]
    io_means    = [np.mean(read_s)] * 3
    total_ms    = [np.mean(total_c), np.mean(total_l), np.mean(total_f)]
    cols        = [C_C, C_L, C_F]

    bar_w = 0.5
    bars_c = ax.bar(methods, comp_means, bar_w, color=cols, edgecolor='white', lw=1.2,
                    zorder=3)
    bars_i = ax.bar(methods, io_means, bar_w, bottom=comp_means, color=C_IO,
                    edgecolor='white', lw=1.2, zorder=3, alpha=.6)

    # label total values
    for i, (bc, iv, tv) in enumerate(zip(bars_c, io_means, total_ms)):
        ax.text(i, bc.get_height() + iv + max(total_ms)*.03,
                f'{tv:.2f} s', ha='center', fontweight='bold', fontsize=10.5)

    # label compute-only percentages
    for i, (bc, tv) in enumerate(zip(bars_c, total_ms)):
        pct = bc.get_height() / tv * 100 if tv > 0 else 0
        ax.text(i, bc.get_height() / 2, f'{pct:.0f}%', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    ax.legend(handles=[
        Patch(facecolor=C_IO, edgecolor='white', label='I/O'),
        Patch(facecolor='#666666', edgecolor='white', label='Computation'),
    ], loc='upper left', framealpha=.9, edgecolor='#CCCCCC', fontsize=9,
        prop={'family': 'serif', 'size': 9})

    ax.set_ylabel('Mean Processing Time per Pair (s)', fontweight='bold', fontsize=11)
    ax.set_title('Average Processing Time Breakdown', fontweight='bold', fontsize=12, pad=8)
    ax.yaxis.grid(True, alpha=.18, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(total_ms) * 1.18)

    fig.tight_layout()
    p = os.path.join(fig_dir, 'time_breakdown.png')
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    log.info("Saved %s", p);  plt.close(fig)

    # ══════════════════════════════════════════════════════════
    #  FIGURE 3 – RMSE bars  +  improvement
    # ══════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # 3a – RMSE per pair
    ax1.bar(x - w, const_mm, w, color=C_C, edgecolor='white', lw=.3, label='Constant')
    ax1.bar(x,      lin_mm,   w, color=C_L, edgecolor='white', lw=.3, label='Linear')
    ax1.bar(x + w,  full_mm,  w, color=C_F, edgecolor='white', lw=.3, label='AINA')

    for i in range(n):
        if full_mm[i] < const_mm[i] * 0.95 and not np.isnan(full_mm[i]):
            ax1.text(i + w, full_mm[i], f'{full_mm[i]:.1f}',
                     ha='center', va='bottom', fontsize=6, fontweight='bold', color=C_F)

    ax1.set_ylabel('WRMSE (mm/yr)')
    ax1.set_title('Pair Adjustment Quality (WRMSE)', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', ncol=3, framealpha=.9, edgecolor='#CCC')
    ax1.yaxis.grid(True, alpha=.18, zorder=0)
    ax1.set_axisbelow(True)

    # 3b – improvement %
    ax2.bar(x - w/2, imp_l, w, color=C_L, edgecolor='white', lw=.3, label='Linear')
    ax2.bar(x + w/2, imp_f, w, color=C_F, edgecolor='white', lw=.3, label='AINA')
    ax2.axhline(y=0, color='#333333', lw=.7, zorder=2)
    # annotate significant improvements
    for i in range(n):
        if imp_f[i] > 20:
            ax2.text(i + w/2, imp_f[i] + 1, f'{imp_f[i]:.0f}%',
                     ha='center', fontsize=6.5, fontweight='bold', color=C_F)

    ax2.set_ylabel('Improvement (%)')
    ax2.set_xlabel('Image Pair')
    ax2.set_title('RMSE Reduction Relative to Constant', fontweight='bold', fontsize=12)
    ax2.set_xticks(x);  ax2.set_xticklabels(label_pairs, rotation=45, ha='right')
    ax2.legend(loc='upper left', framealpha=.9, edgecolor='#CCC')
    ax2.yaxis.grid(True, alpha=.18, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    p = os.path.join(fig_dir, 'rmse_quality.png')
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    log.info("Saved %s", p);  plt.close(fig)

    # ══════════════════════════════════════════════════════════
    #  FIGURE 4 – Scatter + Model Selection
    # ══════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 4a – Const vs Full RMSE scatter
    ok = ~(np.isnan(const_mm) | np.isnan(full_mm))
    ax1.scatter(const_mm[ok], full_mm[ok], c=[C_F if df['f_model'].values[i]=='linear' else C_C
               for i in range(n) if ok[i]], s=55, alpha=.75, edgecolors='white', lw=.8, zorder=5)
    mxx = max(np.max(const_mm[ok]), np.max(full_mm[ok])) * 1.12
    ax1.plot([0, mxx], [0, mxx], '--', color='gray', alpha=.5, lw=1.3, zorder=1)
    ax1.set(xlim=(0, mxx), ylim=(0, mxx));  ax1.set_aspect('equal')
    ax1.set_xlabel('Constant RMSE (mm/yr)')
    ax1.set_ylabel('AINA RMSE (mm/yr)')
    ax1.set_title('RMSE: Constant and AINA', fontweight='bold', fontsize=12)
    # per-pair labels for outliers
    for i in range(n):
        if ok[i] and const_mm[i] > np.percentile(const_mm[ok], 70):
            ax1.annotate(label_pairs[i], (const_mm[i], full_mm[i]),
                         textcoords='offset points', xytext=(4, 4), fontsize=6.5, alpha=.7)
    legend_el = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_C, markersize=7, label='Const chosen'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_F, markersize=7, label='Linear chosen'),
    ]
    ax1.legend(handles=legend_el, fontsize=8, loc='lower right')

    # 4b – Model selection pie
    ax2.pie([n_const, n_linear],
            labels=[f'Constant ({n_const})', f'Linear ({n_linear})'],
            autopct='%1.1f%%', colors=[C_C, C_L], startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'},
            explode=(0, 0.05) if n_linear > 0 else (0, 0))
    ax2.set_title('Model Selection in AINA', fontweight='bold', pad=12)

    fig.tight_layout()
    p = os.path.join(fig_dir, 'scatter_selection.png')
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    log.info("Saved %s", p);  plt.close(fig)


def save_summary(df, res_dir):
    const_mm = df['c_rmse'].values * 1000
    lin_mm   = df['l_rmse'].values * 1000
    full_mm  = df['f_rmse'].values * 1000
    n_pairs  = len(df)
    n_c      = int(np.sum(df['f_model'] == 'const'))
    n_l      = int(np.sum(df['f_model'] == 'linear'))

    # total times including I/O
    read_s  = df['read_sec'].values if 'read_sec' in df.columns else np.zeros(n_pairs)
    total_c = read_s + df['c_time_s'].values
    total_l = read_s + df['l_time_s'].values
    total_f = read_s + df['f_time_s'].values

    lines = [
        "=" * 60,
        "EFFICIENCY COMPARISON  —  SUMMARY",
        "=" * 60,
        f"Total pairs processed: {n_pairs}",
        "",
        f"{'Metric':<32}{'Const':>10}{'Linear':>10}{'Full':>10}",
        "-" * 62,
        f"{'Avg RMSE (mm/yr)':<32}{np.nanmean(const_mm):>10.3f}"
        f"{np.nanmean(lin_mm):>10.3f}{np.nanmean(full_mm):>10.3f}",
        f"{'Median RMSE (mm/yr)':<32}{np.nanmedian(const_mm):>10.3f}"
        f"{np.nanmedian(lin_mm):>10.3f}{np.nanmedian(full_mm):>10.3f}",
        f"{'Std RMSE (mm/yr)':<32}{np.nanstd(const_mm):>10.3f}"
        f"{np.nanstd(lin_mm):>10.3f}{np.nanstd(full_mm):>10.3f}",
        f"{'Min RMSE (mm/yr)':<32}{np.nanmin(const_mm):>10.3f}"
        f"{np.nanmin(lin_mm):>10.3f}{np.nanmin(full_mm):>10.3f}",
        f"{'Max RMSE (mm/yr)':<32}{np.nanmax(const_mm):>10.3f}"
        f"{np.nanmax(lin_mm):>10.3f}{np.nanmax(full_mm):>10.3f}",
        "",
        f"{'Avg compute (ms)':<32}{np.mean(df['c_time_s'])*1e3:>10.2f}"
        f"{np.mean(df['l_time_s'])*1e3:>10.2f}{np.mean(df['f_time_s'])*1e3:>10.2f}",
        f"{'Avg total (I/O+comp, s)':<32}{np.mean(total_c):>10.3f}"
        f"{np.mean(total_l):>10.3f}{np.mean(total_f):>10.3f}",
        f"{'Total runtime (s)':<32}{np.sum(total_c):>10.1f}"
        f"{np.sum(total_l):>10.1f}{np.sum(total_f):>10.1f}",
        f"{'  of which I/O (s)':<32}{np.sum(read_s):>10.1f}"
        f"{np.sum(read_s):>10.1f}{np.sum(read_s):>10.1f}",
        "",
        f"RMSE reduction vs Const:  Linear = "
        f"{np.nanmean((const_mm-lin_mm)/(const_mm+1e-9)*100):.1f}%   Full = "
        f"{np.nanmean((const_mm-full_mm)/(const_mm+1e-9)*100):.1f}%",
        f"Speed factor vs Full:     Const = "
        f"{np.sum(total_f)/np.sum(total_c):.1f}x   Linear = "
        f"{np.sum(total_f)/np.sum(total_l):.1f}x",
        "",
        f"Model selection (Full):   Constant = {n_c}/{n_pairs} "
        f"({n_c/n_pairs*100:.0f}%)   Linear = {n_l}/{n_pairs} "
        f"({n_l/n_pairs*100:.0f}%)",
        "=" * 60,
    ]
    txt = '\n'.join(lines)
    pa = os.path.join(res_dir, 'summary.txt')
    with open(pa, 'w') as f:
        f.write(txt)
    log.info("Saved %s", pa)
    print('\n' + txt)

    csv_p = os.path.join(res_dir, 'comparison_results.csv')
    df.to_csv(csv_p, index=False, float_format='%.8f')
    log.info("Saved %s", csv_p)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def read_pair_overlap(img1, img2):
    """Read overlapping region from two GeoTIFF images, same logic as 1224_new2.py."""
    with rasterio.open(img1['vel']) as s1, rasterio.open(img2['vel']) as s2, \
         rasterio.open(img1['coh']) as c1,  rasterio.open(img2['coh']) as c2:

        xl = max(s1.bounds.left,   s2.bounds.left)
        xr = min(s1.bounds.right,  s2.bounds.right)
        yb = max(s1.bounds.bottom, s2.bounds.bottom)
        yt = min(s1.bounds.top,    s2.bounds.top)
        if xr <= xl or yt <= yb:
            return None

        w = int((xr - xl) / s1.res[0])
        h = int((yt - yb) / s1.res[1])
        if w < 3 or h < 3:
            return None

        trans = from_origin(xl, yt, s1.res[0], s1.res[1])
        prof  = dict(height=h, width=w, transform=trans, crs=s1.crs)

        v1_a = core.reproject_to_grid(s1, prof)
        v2_a = core.reproject_to_grid(s2, prof)
        c1_a = np.nan_to_num(core.reproject_to_grid(c1, prof))
        c2_a = np.nan_to_num(core.reproject_to_grid(c2, prof))

    # mask cascade – same as same_track mode in 1224_new2.py
    mask = None
    for lvl in range(4):
        m = ((c1_a > 0.1) & (c2_a > 0.1)) if lvl == 3 else core.get_adaptive_mask(c1_a, c2_a, lvl)[0]
        if int(np.sum(m)) >= 400:
            mask = m
            break
    if mask is None:
        return None
    mask &= np.isfinite(v1_a) & np.isfinite(v2_a)
    nc = int(np.sum(mask))
    if nc < 400:
        return None

    ix = np.flatnonzero(mask)
    v1 = v1_a.ravel()[ix].astype(np.float64)
    v2 = v2_a.ravel()[ix].astype(np.float64)
    w1 = np.clip(c1_a.ravel()[ix], eps_w, 1.0) ** 2
    w2 = np.clip(c2_a.ravel()[ix], eps_w, 1.0) ** 2

    return v1, v2, w1, w2, nc


def process_year(year):
    """Run full comparison for one year, output into yr_<year>/ subfolder."""
    yr_dir    = os.path.join(SCRIPT_DIR, f'yr_{year}')
    fig_dir   = os.path.join(yr_dir, 'figures')
    res_dir   = os.path.join(yr_dir, 'results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # ── same input pattern as 1224_new2.py ──────────────────
    images = [
        {'vel': fr'E:\shanxi\{year}\{year}_113_126\vel_{year}_113_126.tif',
         'coh': fr'E:\shanxi\{year}\{year}_113_126\coh_{year}_113_126.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_113_121\vel_{year}_113_121.tif',
         'coh': fr'E:\shanxi\{year}\{year}_113_121\coh_{year}_113_121.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_113_116\vel_{year}_113_116.tif',
         'coh': fr'E:\shanxi\{year}\{year}_113_116\coh_{year}_113_116.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_113_111\vel_{year}_113_111.tif',
         'coh': fr'E:\shanxi\{year}\{year}_113_111\coh_{year}_113_111.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_11_121\vel_{year}_11_121.tif',
         'coh': fr'E:\shanxi\{year}\{year}_11_121\coh_{year}_11_121.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_11_116\vel_{year}_11_116.tif',
         'coh': fr'E:\shanxi\{year}\{year}_11_116\coh_{year}_11_116.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_11_111\vel_{year}_11_111.tif',
         'coh': fr'E:\shanxi\{year}\{year}_11_111\coh_{year}_11_111.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_40_127\vel_{year}_40_127.tif',
         'coh': fr'E:\shanxi\{year}\{year}_40_127\coh_{year}_40_127.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_40_122\vel_{year}_40_122.tif',
         'coh': fr'E:\shanxi\{year}\{year}_40_122\coh_{year}_40_122.tif'},
        {'vel': fr'E:\shanxi\{year}\{year}_40_117\vel_{year}_40_117.tif',
         'coh': fr'E:\shanxi\{year}\{year}_40_117\coh_{year}_40_117.tif'},
    ]
    for i, d in enumerate(images):
        d['id'] = i
        # extract frame ID from filename, e.g. "vel_17_113_126.tif" -> "113_126"
        m = re.search(r'_(\d+_\d+)\.tif$', d['vel'])
        d['frame'] = m.group(1) if m else str(i)

    first_file = images[0]['vel']
    if not os.path.exists(first_file):
        log.warning("Year %d: data not found at %s — skipped", year, first_file)
        return

    log.info("=" * 60)
    log.info("Year %d  —  processing", year)
    log.info("=" * 60)

    records = []
    pair_idx = 0
    n_images = len(images)
    for i in range(n_images):
        for j in range(i + 1, n_images):
            log.info("Reading pair %d-%d …", i, j)
            t0 = time.time()
            overlap = read_pair_overlap(images[i], images[j])
            t_read = time.time() - t0
            if overlap is None:
                log.warning("  → skipped (insufficient overlap)")
                continue
            v1, v2, w1, w2, nc = overlap
            rec = process_one_pair(v1, v2, w1, w2, pair_idx)
            rec['pair'] = f'{images[i]["frame"]}\n{images[j]["frame"]}'
            rec['n_points'] = nc
            rec['read_sec'] = t_read
            records.append(rec)
            log.info("%d-%d  |  %d pts  |  read %.2fs  |  RMSE c=%.4f l=%.4f f=%.4f  |  %.2f/%.2f/%.2f ms  |  model=%s",
                     i, j, nc, t_read, rec['c_rmse'], rec['l_rmse'], rec['f_rmse'],
                     rec['c_time_s']*1e3, rec['l_time_s']*1e3, rec['f_time_s']*1e3, rec['f_model'])
            pair_idx += 1

    if not records:
        log.warning("Year %d: no valid pairs.", year)
        return

    df = pd.DataFrame(records)
    make_figures(df, fig_dir)
    save_summary(df, res_dir)
    log.info("Year %d done — outputs in %s", year, yr_dir)


def main():
    years = [17, 18, 19, 2123]
    for y in years:
        process_year(y)
    log.info("All years complete — outputs in %s", SCRIPT_DIR)


if __name__ == '__main__':
    main()
