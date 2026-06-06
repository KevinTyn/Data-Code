import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# =================================================================
# 【核心控制面板】 - 这里的参数决定了最终的间距美感
# =================================================================

# 0. 中心空洞控制
CENTER_HOLE = 1.5

# 1. 柱体与间距
BAR_WIDTH_VAL = 0.8  # 稍微调细一点（如0.8），会让居中效果在视觉上更明显
INNER_GAP = 0.5  # 指标内部的缝隙
OUTER_GAP = 4.0  # 簇与簇之间的总留白（此时虚线会平分这个留白）

# 2. 径向距离控制
MAX_VAL_LIMIT = 15.5
BASE_INNER = 8.1
BASE_OUTER = 10.5
LABEL_RADIAL_POS = 9.6

# 3. 装饰线与颜色
OUTER_LINE_COLOR = '#F1EDEA'
OUTER_LINE_GAP = -0.1
OUTER_LINE_THICK = 1
OUTER_LINE_ALPHA = 0.6

YEAR_COLORS = ['#9F87B9', '#B33648', '#DB7368']
BG_COLORS = ['#F2EFF6', '#F5F2E9', '#FBF7F4']
SECTOR_ALPHA = 1.0

# 4. 刻度与虚线
TICKS = [1, 3, 5, 7, 9]
GRID_ALPHA = 0.3
DIVIDER_ALPHA = 1.0  # 簇间分割虚线的透明度

# =================================================================
# 【绘图逻辑区】 - 已修复居中偏移算法
# =================================================================

scene_labels = ["111_116", "116_121", "117_122", "122_127", "111_116", "116_121", "121_126", "", ""]

# ---> [新增] 外圈线上需要显示的文字内容（9个场景对应9个标签） <---
outer_labels = ["P11", "P11", "P40", "P40", "P113", "P113", "P113", "P11_P113", "P113_P40"]

rmse_m = np.array([[7.50, 7.42, 7.39], [6.68, 6.71, 6.46], [8.01, 8.13, 7.91], [6.66, 6.78, 6.52], [6.97, 6.84, 7.12],
                   [6.56, 6.47, 6.64], [7.34, 7.21, 7.18], [5.36, 5.43, 5.27], [5.05, 5.17, 4.93]])
mae_m = np.array([[5.49, 5.38, 5.44], [4.90, 4.82, 4.73], [6.54, 6.63, 6.42], [4.74, 4.81, 4.67], [4.95, 4.88, 5.08],
                  [4.68, 4.54, 4.76], [4.91, 4.79, 4.84], [4.14, 4.28, 4.09], [4.17, 4.23, 4.06]])
std_m = np.array([[7.48, 7.36, 7.27], [6.68, 6.52, 6.43], [7.70, 7.83, 7.58], [6.55, 6.67, 6.41], [5.77, 5.64, 5.86],
                  [6.53, 6.41, 6.62], [7.06, 6.94, 6.88], [5.33, 5.47, 5.19], [3.41, 3.58, 3.24]])
matrices = [rmse_m, mae_m, std_m]
metrics = ['RMSE', 'MAE', 'STD']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

num_scenes, num_metrics, num_times = 9, 3, 3

# 精确计算每个场景占用的总宽度
content_width = (num_metrics * num_times + (num_metrics - 1) * INNER_GAP)
scene_width = content_width + OUTER_GAP
total_slots = num_scenes * scene_width


def get_ang(slot): return (slot / total_slots) * 2 * np.pi


fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# --- 关键修改：偏移 0.5 以抵消 bar 的中心对齐模式 ---
curr = OUTER_GAP / 2 + 0.5
scene_dividers = []

for s in range(num_scenes):
    # 记录当前簇的起始槽位（不含半个柱宽的偏移）
    content_start = curr - 0.5

    # 分割虚线：精确位于 Outer Gap 的中心
    scene_dividers.append(get_ang(content_start - OUTER_GAP / 2))

    for m_idx, m_name in enumerate(metrics):
        m_start_val = curr

        # 绘制柱子
        for t in range(num_times):
            angle = get_ang(curr)
            ax.bar(angle, matrices[m_idx][s, t],
                   width=(2 * np.pi / total_slots) * BAR_WIDTH_VAL,
                   color=YEAR_COLORS[t], edgecolor='none', zorder=3)
            curr += 1

        # 背景扇区：从当前组第一个柱子的左边缘到最后一个柱子的右边缘
        m_ang_range = np.linspace(get_ang(m_start_val - 0.5), get_ang(curr - 0.5), 50)
        ax.fill_between(m_ang_range, 0, BASE_INNER, color=BG_COLORS[m_idx], alpha=SECTOR_ALPHA, zorder=0)

        if m_idx < 2: curr += INNER_GAP

    # 场景装饰
    content_end = curr - 0.5
    s_mid_ang = get_ang((content_start + content_end) / 2)
    s_ang_range = np.linspace(get_ang(content_start - 0.2), get_ang(content_end + 0.2), 100)

    # 外边缘强化线
    line_s, line_e = BASE_OUTER + OUTER_LINE_GAP, BASE_OUTER + OUTER_LINE_GAP + OUTER_LINE_THICK
    ax.fill_between(s_ang_range, line_s, line_e, color=OUTER_LINE_COLOR, alpha=OUTER_LINE_ALPHA, zorder=2)

    # 场景标签
    s_rot = np.degrees(s_mid_ang)
    sf_rot = s_rot + 180 if 90 < s_rot < 270 else s_rot
    ax.text(s_mid_ang, LABEL_RADIAL_POS, scene_labels[s], fontsize=16, fontweight='bold',
            ha='center', va='center', rotation=-sf_rot, zorder=6)

    # ---> [新增] 在外边缘强化线上绘制文字 <---
    outer_text_r = (line_s + line_e) / 2  # 计算外圈线的径向中点
    ax.text(s_mid_ang, outer_text_r, outer_labels[s], fontsize=16, fontweight='bold',
            ha='center', va='center', rotation=-sf_rot, zorder=7,
            color='#377EB6')  # <--- 在这里换颜色，我这里配了一个和柱子同色系的紫色

    curr += OUTER_GAP

# 绘制场景分割虚线
for div_ang in scene_dividers:
    ax.plot([div_ang, div_ang], [0, BASE_INNER], color='#AAAAAA', lw=0.8, ls=(0, (5, 5)), zorder=1)

# 背景刻度
label_angle = scene_dividers[0]
for r in TICKS:
    ax.plot(np.linspace(0, 2 * np.pi, 200), [r] * 200, color='#888888', lw=0.5, ls='--', alpha=0.3)
    ax.text(label_angle, r, f'{r}', color='#333333', fontsize=14, fontweight='bold',
            ha='center', va='center', zorder=5)

ax.set_axis_off()
ax.set_ylim(-CENTER_HOLE, MAX_VAL_LIMIT)

plt.tight_layout()

# 添加下面这行代码来保存高分辨率图片
plt.savefig('polar_chart.png', dpi=600, bbox_inches='tight')

plt.show()