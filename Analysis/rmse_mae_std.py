import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================
# 0. 输出目录 (桌面)
# ============================
save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "shanxi_radar")
os.makedirs(save_dir, exist_ok=True)

# ============================
# 1. 学术化全局设置
# ============================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24

# ============================
# 2. 数据准备与轨道标记
# ============================
scene_labels = ["111_116", "116_121", "117_122", "122_127", "111_116", "116_121", "121_126", "11_113", "113_40"]

formatted_labels = []
for i, label in enumerate(scene_labels):
    if i < 7:
        formatted_labels.append(f"{label}")
    else:
        formatted_labels.append(f"{label}")

# 所有数据集 (第4-7组也保留但本次不循环)
datasets = [
    {
        'rmse_ori': np.array([20.14, 15.62, 16.47, 14.88, 19.36, 16.02, 14.78, 15.19, 19.82]),
        'rmse_cor': np.array([18.93, 14.28, 15.00, 14.11, 18.17, 13.87, 13.50, 13.22, 18.74]),
        'mae_ori':  np.array([15.82, 12.07, 13.66, 11.24, 16.89, 10.43, 11.81, 12.14, 14.33]),
        'mae_cor':  np.array([14.18, 11.58, 12.86, 10.93, 15.12,  9.97, 10.13, 10.79, 15.53]),
        'std_ori':  np.array([11.06, 10.33,  9.48, 10.12, 10.09, 11.27, 10.88, 10.76, 11.41]),
        'std_cor':  np.array([18.84, 14.21, 14.93, 14.04, 18.08, 13.80, 13.43, 13.15, 18.65]),
    },
    {
        'rmse_ori': np.array([16.97, 14.89, 15.43, 14.23, 13.74, 15.13, 14.41, 14.67, 15.91]),
        'rmse_cor': np.array([13.58,  9.33, 10.02, 10.49, 11.12,  8.72,  9.07, 10.18, 10.41]),
        'mae_ori':  np.array([13.02, 12.11, 13.27, 11.42, 11.88, 10.96, 11.93, 12.02, 12.86]),
        'mae_cor':  np.array([12.73,  7.00,  8.47,  9.18,  8.21,  8.12,  8.58,  9.57, 10.09]),
        'std_ori':  np.array([13.97, 13.42, 11.00, 10.00, 13.22, 11.00, 12.49, 14.17, 12.39]),
        'std_cor':  np.array([13.51,  9.28,  9.97, 10.44, 11.06,  8.68,  9.02, 10.13, 10.36]),
    },
]
#############################################################################精度验证
# 17
# rmse_ori = np.array([14.64, 10.82, 14.17, 11.26, 12.53, 15.03, 10.76, 9.73, 10.50])
# rmse_cor = np.array([10.43, 7.56, 10.29, 8.87, 9.67, 10.60, 8.22, 7.00, 8.36])
# mae_ori = np.array([11.52, 8.85, 10.77, 7.62, 10.61, 9.75, 7.84, 7.79, 6.93])
# mae_cor = np.array([7.51, 5.56, 7.74, 6.46, 7.19, 7.63, 6.05, 5.26, 6.09])
# std_ori = np.array([10.04, 7.50, 9.76, 5.07, 9.42, 6.63, 6.04, 6.39, 4.91])
# std_cor = np.array([10.38, 7.52, 10.24, 8.83, 9.62, 10.55, 8.18, 6.96, 8.32])

# 18
# rmse_ori = np.array([14.30, 7.03, 16.04, 11.99, 29.03, 12.54, 15.94, 10.60, 13.47])
# rmse_cor = np.array([8.62, 4.38, 6.90, 7.16, 6.94, 9.78, 7.29, 5.02, 7.81])
# mae_ori = np.array([12.08, 5.98, 12.61, 9.30, 27.44, 8.23, 14.61, 9.54, 10.75])
# mae_cor = np.array([6.35, 3.25, 4.87, 5.09, 4.91, 6.38, 5.09, 3.73, 5.80])
# std_ori = np.array([11.17, 4.38, 13.07, 8.81, 10.34, 12.37, 6.77, 4.92, 9.91])
# std_cor = np.array([8.60, 4.38, 6.86, 6.78, 6.93, 9.72, 7.26, 4.87, 7.24])

# 19
# rmse_ori = np.array([22.17,30.19,15.65,12.91,37.04,23.75,9.24,16.84,8.38])
# rmse_cor = np.array([7.42,6.01,8.31,6.20,7.91,7.60,3.93,5.43,6.01])
# mae_ori = np.array([19.50,29.59,14.11,11.15,35.22,22.80,7.24,16.31,6.80])
# mae_cor = np.array([5.14,4.79,6.58,4.68,6.14,5.35,2.68,4.51,5.23])
# std_ori = np.array([12.78,5.99,7.11,7.54,11.52,7.06,7.82,4.22,6.45])
# std_cor = np.array([7.01,5.99,7.11,5.49,7.60,7.06,3.88,3.96,3.61])

#############################################################################0.75/equal没有展示
#0.75/equal-2017
# rmse_ori = np.array([16.29,23.62,21.47,11.59,11.67,24.52,20.12,18.27,19.03])
# rmse_cor = np.array([13.11,24.07,22.99,11.11,9.17,23.26,23.87,20.08,22.32])
# mae_ori = np.array([10.57,15.98,15.65,9.40,7.82,15.49,13.16,12.71,16.23])
# mae_cor = np.array([7.20,17.40,17.42,8.85,5.74,14.48,12.30,17.76,17.21])
# std_ori = np.array([16.28,23.53,21.27,11.59,11.20,24.73,18.11,18.21,15.37])
# std_cor = np.array([13.10,23.21,22.99,10.91,9.17,23.18,16.87,14.02,16.44])

#0.75/equal-2018
# rmse_ori = np.array([20.99, 14.86, 13.98, 18.82, 21.33, 16.91, 25.23, 20.81, 17.77])
# rmse_cor = np.array([14.80, 13.97, 20.49, 15.58, 9.18, 14.90, 20.64, 11.25, 11.14])
# mae_ori = np.array([12.35, 12.36, 13.42, 15.66, 14.36, 13.18, 14.34, 10.18, 10.59])
# mae_cor = np.array([7.94, 8.86, 13.59, 6.03, 10.82, 10.78, 11.66, 10.71, 9.12])
# std_ori = np.array([16.82, 24.07, 24.86, 17.99, 19.89, 12.00, 15.13, 16.64, 23.65])
# std_cor = np.array([12.19, 15.92, 23.98, 16.50, 18.00, 10.10, 21.42, 17.84, 16.91])

#0.75/equal-2019
# rmse_ori = np.array([20.12, 23.97, 18.71, 20.24, 19.59, 21.28, 14.50, 13.05, 14.64])
# rmse_cor = np.array([20.38, 19.12, 22.80, 17.33, 15.80, 23.08, 17.59, 21.58, 14.39])
# mae_ori = np.array([14.03, 14.62, 10.32, 8.86, 16.65, 10.27, 10.54, 14.91, 15.18])
# mae_cor = np.array([12.44, 10.83, 10.82, 9.86, 14.40, 9.39, 11.38, 15.84, 9.89])
# std_ori = np.array([21.20, 20.87, 16.43, 13.93, 16.46, 11.84, 18.75, 21.89, 22.60])
# std_cor = np.array([12.81, 17.03, 22.54, 13.58, 22.14, 10.23, 17.73, 17.39, 22.93])

##########################################################################消融实验
# constant+linear-2017
# rmse_ori = np.array([20.14, 15.62, 16.47, 14.88, 19.36, 16.02, 14.78, 15.19, 19.82])
# rmse_cor = np.array([18.93, 14.28, 15.00, 14.11, 18.17, 13.87, 13.50, 13.22, 18.74])
# mae_ori  = np.array([15.82, 12.07, 13.66, 11.24, 16.89, 10.43, 11.81, 12.14, 14.33])
# mae_cor  = np.array([14.18, 11.58, 12.86, 10.93, 15.12,  9.97, 10.13, 10.79, 15.53])
# std_ori  = np.array([11.06, 10.33,  9.48, 10.12, 10.09, 11.27, 10.88, 10.76, 11.41])
# std_cor  = np.array([18.84, 14.21, 14.93, 14.04, 18.08, 13.80, 13.43, 13.15, 18.65])
# igg+ransac-2017
# rmse_ori = np.array([16.97, 14.89, 15.43, 14.23, 13.74, 15.13, 14.41, 14.67, 15.91])
# rmse_cor = np.array([13.58,  9.33, 10.02, 10.49, 11.12,  8.72,  9.07, 10.18, 10.41])
# mae_ori  = np.array([13.02, 12.11, 13.27, 11.42, 11.88, 10.96, 11.93, 12.02, 12.86])
# mae_cor  = np.array([12.73,  7.00,  8.47,  9.18,  8.21,  8.12,  8.58,  9.57, 10.09])
# std_ori  = np.array([13.97, 13.42, 11.00, 10.00, 13.22, 11.00, 12.49, 14.17, 12.39])
# std_cor  = np.array([13.51,  9.28,  9.97, 10.44, 11.06,  8.68,  9.02, 10.13, 10.36])


# rmse_ori= np.array([15.71, 15.13, 16.82, 20.76, 23.66, 18.21, 20.58, 19.34, 19.92])
# rmse_cor = np.array([19.57, 22.41, 9.58, 21.08, 12.29, 11.45, 17.84, 9.79, 23.13])
#
# mae_ori = np.array([13.35, 14.35, 6.99, 13.01, 12.62, 8.47, 8.41, 14.06, 10.14])
# mae_cor = np.array([14.44, 12.51, 10.19, 17.71, 16.54, 9.75, 5.35, 9.01, 10.32])
#
# std_ori = np.array([21.15, 25.66, 15.73, 22.04, 19.41, 20.94, 12.55, 16.39, 13.94])
# std_cor = np.array([13.68, 16.40, 18.86, 9.87, 10.27, 13.33, 18.78, 21.75, 17.04])
# ============================
# 3. 绘图与保存
# ============================
def save_accuracy_plots(dataset, idx):
    N = len(scene_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    color_ori = '#555555'
    color_cor = '#084594'

    metrics = [
        ('RMSE (m)', dataset['rmse_ori'], dataset['rmse_cor'], 'RMSE_Result'),
        ('MAE (m)',  dataset['mae_ori'],  dataset['mae_cor'],  'MAE_Result'),
        ('STD (m)',  dataset['std_ori'],  dataset['std_cor'],  'STD_Result')
    ]

    for title, ori, cor, filename in metrics:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        v_ori = np.append(ori, ori[0])
        v_cor = np.append(cor, cor[0])

        ax.plot(angles_closed, v_ori, color=color_ori, linewidth=2.5, linestyle='--', label='Before Correction')
        ax.fill(angles_closed, v_ori, color=color_ori, alpha=0.1)

        ax.plot(angles_closed, v_cor, color=color_cor, linewidth=2.5, linestyle='-', label='After Correction')
        ax.fill(angles_closed, v_cor, color=color_cor, alpha=0.3)

        max_val = max(ori.max(), cor.max())
        base_r_offset = max_val * 0.10

        for i, (angle, val_cor, val_ori) in enumerate(zip(angles, cor, ori)):
            if val_cor < val_ori:
                r_pos = max(0, val_cor - base_r_offset)
            else:
                r_pos = val_cor + base_r_offset

            angle_offset = 0.06 if i % 2 == 0 else -0.06
            final_angle = angle + angle_offset

            ax.text(final_angle, r_pos, f'{val_cor:.2f}',
                    ha='center', va='center',
                    fontsize=18,
                    fontfamily='Times New Roman',
                    color='black')

        ax.set_xticks(angles)
        ax.set_xticklabels(formatted_labels, fontsize=24, fontweight='bold')
        ax.spines['polar'].set_visible(False)
        ax.grid(True, axis='both', color='#888888', linestyle='--')
        ax.set_ylim(0, max_val * 1.20)

        ax.set_title(f"{title}  (Group #{idx})", size=24, pad=45, fontweight='bold', family='Times New Roman')

        ax.legend(loc='lower center', ncol=2, frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.15))

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"G{idx}_{filename}.png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
        print(f"已保存: {save_path}")

        plt.close()

# ============================
# 4. 循环输出 #5
# ============================
for idx, ds in enumerate(datasets, start=1):
    save_accuracy_plots(ds, idx)