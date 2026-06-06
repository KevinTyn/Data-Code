import matplotlib.pyplot as plt
import numpy as np

# 1. 准备经计算后的精确数据
regions = ['Path 11 – Path 113\n(Inter-track)', 'Path 113 – Path 40\n(Inter-track)']
filtered_out = [45.95, 37.62]  # 被过滤掉的百分比
retained = [100 - f for f in filtered_out]  # 保留的百分比

# 2. 设置学术图表样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)

bar_width = 0.45

# 3. 绘制100%堆叠水平条形图 (使用高级学术配色：暗红/深灰蓝)
color_filtered = '#d9534f'  # Muted Crimson (表示数据丢失/警告)
color_retained = '#34495e'  # Dark Slate (表示稳定保留)

b1 = ax.barh(regions, filtered_out, bar_width, label='Filtered Out (Fixed 0.75)', color=color_filtered, alpha=0.9)
b2 = ax.barh(regions, retained, bar_width, left=filtered_out, label='Retained Pixels', color=color_retained, alpha=0.9)

# 4. 在柱状图内部自动添加百分比标签
for bar in b1:
    width = bar.get_width()
    ax.text(width / 2, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%',
            va='center', ha='center', color='white', fontweight='bold', fontsize=10)

for bar in b2:
    width = bar.get_width()
    left = bar.get_x()
    ax.text(left + width / 2, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%',
            va='center', ha='center', color='white', fontweight='bold', fontsize=10)

# 5. 优化细节与美化
ax.set_xlabel('Percentage of Total Pixels (%)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_xlim(0, 100)
ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
ax.set_axisbelow(True)

# 移除上方和右方的边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 优化标签刻度
ax.tick_params(axis='both', labelsize=10)

# 6. 图例放置在下方
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=10)

plt.tight_layout()
plt.show()