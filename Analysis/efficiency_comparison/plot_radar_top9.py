# -*- coding: utf-8 -*-
"""Radar chart: top-9 worst pairs for yr_18 and yr_19, comparing Const / Linear / AINA RMSE."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── style ──────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24

BASE    = os.path.dirname(os.path.abspath(__file__))
DESKTOP = os.path.join(os.path.expanduser('~'), 'Desktop')

# ── build one radar chart ──────────────────────────
def plot_radar(df, year, out_path, top_n=9):
    df = df.copy()
    df['const_mm'] = df['c_rmse'] * 1000
    worst = df.nlargest(top_n, 'const_mm')
    labels = worst['pair'].values
    const  = worst['const_mm'].values
    linear = worst['l_rmse'].values * 1000
    aina   = worst['f_rmse'].values * 1000

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.plot(angles_closed, np.append(const, const[0]),
            color='#999999', linewidth=2.5, linestyle='--', label='Constant')
    ax.fill(angles_closed, np.append(const, const[0]), color='#999999', alpha=0.08)

    ax.plot(angles_closed, np.append(linear, linear[0]),
            color='#2E86AB', linewidth=2.5, linestyle='-', label='Linear')
    ax.fill(angles_closed, np.append(linear, linear[0]), color='#2E86AB', alpha=0.10)

    ax.plot(angles_closed, np.append(aina, aina[0]),
            color='#F18F01', linewidth=2.5, linestyle='-', label='AINA')
    ax.fill(angles_closed, np.append(aina, aina[0]), color='#F18F01', alpha=0.15)

    # annotate AINA values
    max_val = max(const.max(), linear.max(), aina.max())
    base_r  = max_val * 0.08
    for i, (angle, val) in enumerate(zip(angles, aina)):
        offset = 0.06 if i % 2 == 0 else -0.06
        r_pos  = max(0, val - base_r)
        ax.text(angle + offset, r_pos, f'{val:.1f}', ha='center', va='center',
                fontsize=16, fontfamily='Times New Roman', color='#333333')

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=18, fontweight='bold')
    ax.spines['polar'].set_visible(False)
    ax.grid(True, axis='both', color='#AAAAAA', linestyle='--', alpha=0.6)
    ax.set_ylim(0, max_val * 1.18)
    ax.set_title(f'Top-9 Worst Pairs — {year}', size=22, pad=40,
                 fontweight='bold', family='Times New Roman')
    ax.legend(loc='lower center', ncol=3, frameon=False, fontsize=14,
              bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ── main ───────────────────────────────────────────
for yr in [18, 19]:
    csv_path = os.path.join(BASE, f'yr_{yr}', 'results', 'comparison_results.csv')
    df = pd.read_csv(csv_path)
    out = os.path.join(DESKTOP, f'radar_top9_{yr}.png')
    plot_radar(df, yr, out)
    print(f'Year {yr} done')
