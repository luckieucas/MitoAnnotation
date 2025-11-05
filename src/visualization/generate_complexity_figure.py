import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ====== 从CSV文件读取数据 ======
csv_file_path = "/projects/weilab/liupeng/dataset/mito/complexity/MitoHard_complexity.csv"
df = pd.read_csv(csv_file_path)

# ====== 设置样式 ======
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
fig, ax = plt.subplots(figsize=(14, 8))

# ====== 色盲友好配色（基于ColorBrewer和IBM的色盲友好方案）======
# 使用橙色系代替红色，蓝色系保持
hard_color = '#E69F00'  # 橙色（色盲友好）
easy_color = '#0173B2'  # 蓝色（色盲友好）

# 按组分类
hard_datasets = df[df["group"] == "hard"]["dataset"].tolist()
easy_datasets = df[df["group"] == "easy"]["dataset"].tolist()

# ====== 添加简单数据集参照区域（DCI 0-0.5, EFI 0.95-1.2）======
# 绘制矩形区域，虚线边框不延伸
from matplotlib.patches import Rectangle
easy_region = Rectangle(
    (0, 0.95),  # 左下角坐标 (x, y)
    0.5,        # 宽度 (DCI: 0 to 0.5)
    0.25,       # 高度 (EFI: 0.95 to 1.2)
    linewidth=2,
    edgecolor='#0173B2',
    facecolor='#0173B2',
    alpha=0.15,
    linestyle='--',
    zorder=0,
    label='Easy dataset region'
)
ax.add_patch(easy_region)

# ====== 绘制hard组（星号 + 橙色）======
for i, dataset in enumerate(hard_datasets):
    subset = df[df["dataset"] == dataset]
    x = subset["dci"].values[0]
    y = subset["efi"].values[0]
    
    ax.scatter(
        x, y,
        c=hard_color, 
        s=1200,
        edgecolor='white', 
        linewidth=2.5,
        label=dataset, 
        alpha=0.95, 
        marker="*",
        zorder=3
    )

# ====== 绘制easy组（圆圈 + 蓝色）======
for i, dataset in enumerate(easy_datasets):
    subset = df[df["dataset"] == dataset]
    x = subset["dci"].values[0]
    y = subset["efi"].values[0]
    
    ax.scatter(
        x, y,
        c=easy_color, 
        s=300,
        edgecolor='white',
        linewidth=2.5,
        label=dataset, 
        alpha=0.95, 
        marker="o",
        zorder=3
    )

# ====== 图表美化 ======
ax.set_title("Mapping Dataset Complexity in DCI–EFI Space", fontsize=18, weight='bold', pad=20)
ax.set_xlabel("DCI", fontsize=18, weight='semibold')
ax.set_ylabel("EFI", fontsize=18, weight='semibold')

# 设置坐标轴范围
ax.set_xlim(0, 4.2)
ax.set_ylim(0.95, 2)

# 设置坐标轴刻度
ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

# 调整刻度字体大小
ax.tick_params(axis='both', labelsize=14)

# ====== 去掉上/右边框，保留左/下边框 ======
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_edgecolor('#666666')
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_edgecolor('#666666')
ax.spines['bottom'].set_linewidth(1.5)

# ====== 优化图例（形状+颜色双编码）======
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='*', color='w', label='Our datasets (hard)',
           markerfacecolor=hard_color, markersize=25, 
           markeredgecolor='white', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', label='Previous datasets (easy)',
           markerfacecolor=easy_color, markersize=18,
           markeredgecolor='white', markeredgewidth=2),
    Rectangle((0, 0), 1, 1, facecolor='#0173B2', edgecolor='#0173B2', 
              alpha=0.15, linestyle='--', linewidth=2,
              label='Easy region')
]

legend = ax.legend(
    handles=legend_elements,
    loc='upper right',
    fontsize=14,
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='gray',
    handletextpad=0.9,
    borderpad=1.2,
    labelspacing=1.0
)

# ====== 添加浅网格样式 ======
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='#CCCCCC')
ax.set_axisbelow(True)

# 设置背景色
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()

# 保存 & 显示
plt.savefig("figs/MitoHard_complexity.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig("figs/MitoHard_complexity.pdf", bbox_inches='tight', facecolor='white')
plt.show()