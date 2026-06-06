import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 全局字体与字号设置 (符合国际顶刊排版标准)
# ==========================================
# 强制使用 Times New Roman 字体
plt.rcParams['font.family'] = 'Times New Roman'
# 让数学公式 (如 $t$, $v_1$) 的字体也匹配衬线体风格
plt.rcParams['mathtext.fontset'] = 'stix'
# 全局基础字号放大
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 22

# ==========================================
# 1. 模拟数据：生成潜变量 t 与观测值 v1, v2
# ==========================================
np.random.seed(42)
x = np.linspace(0, 10, 150)

# 潜变量 t：真实的到底层形变速率曲线
t_latent = 5 * np.sin(x) + 2 * np.cos(2*x)

# 影像 1 观测值 (正向偏移，幅度放大)
a1, b1 = 3.0, 1.2
v1_obs = a1 + b1 * t_latent + np.random.normal(0, 0.5, size=x.shape)

# 影像 2 观测值 (负向偏移，幅度缩小)
a2, b2 = -4.0, 0.8
v2_obs = a2 + b2 * t_latent + np.random.normal(0, 0.5, size=x.shape)

# ==========================================
# 2. 绘制“潜变量映射”示意图 (加大版)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) # 画布适度放大以容纳大字体

# ------------------------------------------
# 左图：空间剖面视图 (Spatial Profile)
# ------------------------------------------
ax1.plot(x, t_latent, color='black', linewidth=3.5, linestyle='-', zorder=5,
         label='Latent Truth $t$ (True Deformation)')

ax1.scatter(x, v1_obs, color='royalblue', alpha=0.5, s=30, label='Image 1 Obs ($v_1$)')
ax1.plot(x, a1 + b1*t_latent, color='royalblue', linewidth=2.5, linestyle='--')

ax1.scatter(x, v2_obs, color='darkorange', alpha=0.5, s=30, label='Image 2 Obs ($v_2$)')
ax1.plot(x, a2 + b2*t_latent, color='darkorange', linewidth=2.5, linestyle='--')

# 绘制关联虚线
for idx in [20, 75, 120]:
    ax1.plot([x[idx], x[idx]], [v2_obs[idx], v1_obs[idx]], color='gray', linestyle=':', alpha=0.8, linewidth=2)
    ax1.plot(x[idx], t_latent[idx], 'ko', markersize=8, zorder=6)

ax1.set_title('Spatial View: Same Latent Truth, Different Observations', fontweight='bold', pad=15)
ax1.set_xlabel('Spatial Position ($X$)')
ax1.set_ylabel('Velocity ($v$)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# ------------------------------------------
# 右图：潜变量映射视图 (Latent Variable Mapping)
# ------------------------------------------
t_range = np.linspace(t_latent.min()-1, t_latent.max()+1, 100)

ax2.scatter(t_latent, v1_obs, color='royalblue', alpha=0.5, s=30)
ax2.plot(t_range, a1 + b1*t_range, color='royalblue', linewidth=3.5,
         label=f'Image 1 Mapping: $v_1 = {a1:.1f} + {b1:.1f}t$')

ax2.scatter(t_latent, v2_obs, color='darkorange', alpha=0.5, s=30)
ax2.plot(t_range, a2 + b2*t_range, color='darkorange', linewidth=3.5,
         label=f'Image 2 Mapping: $v_2 = {a2:.1f} + {b2:.1f}t$')

ax2.plot(t_range, t_range, color='black', linewidth=2.5, linestyle='-.', alpha=0.6,
         label='Ideal Observation: $v = t$')

# 动态映射示例线
demo_t = t_latent[75]
demo_v1 = a1 + b1 * demo_t
demo_v2 = a2 + b2 * demo_t
ax2.plot([demo_t, demo_t], [demo_v2, demo_v1], color='red', linestyle='--', linewidth=2)
ax2.plot(demo_t, demo_t, 'ko', markersize=10, zorder=5)
ax2.annotate('Latent Variable $t_i$', xy=(demo_t, demo_t), xytext=(demo_t+0.5, demo_t-2.5),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=16, fontweight='bold')
ax2.plot(demo_t, demo_v1, 'bo', markersize=8)
ax2.plot(demo_t, demo_v2, 'o', color='darkorange', markersize=8)

ax2.set_title('Latent Space View: Projection via Parameters ($a, b$)', fontweight='bold', pad=15)
ax2.set_xlabel('Latent True Velocity ($t$)')
ax2.set_ylabel('Observed Velocity ($v_1, v_2$)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Latent Variable Model in InSAR Co-registration', fontweight='bold', y=1.02)
plt.subplots_adjust(wspace=0.25) # 增加左右子图间距，防止大字体互相遮挡
plt.savefig('latent_variable_mapping_times_new_roman.png', dpi=300, bbox_inches='tight')
# plt.show()