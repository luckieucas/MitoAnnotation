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

fig, ax = plt.subplots(figsize=(12, 7))

# ====== 定义更鲜明的颜色方案 ======
# Hard组：使用色轮上均匀分布的高对比度颜色（支持最多11个样本）
# 每个颜色在色相、饱和度、明度上都有显著差异
hard_colors = [
    '#E53935',  # 鲜红色 (红)
    '#FF6F00',  # 深橙色 (橙)
    '#FBC02D',  # 亮黄色 (黄) - 新
    '#7CB342',  # 草绿色 (绿) - 新
    '#00ACC1',  # 青色 (青) - 新
    '#5E35B1',  # 深紫色 (紫) - 新
    '#D81B60',  # 玫红色 (品红)
    '#6D4C41',  # 深棕色 (棕)
    '#FF5722',  # 火红橙 (红橙)
    '#EC407A',  # 粉红色 (粉) - 新
    '#8E24AA'   # 紫罗兰 (紫罗兰) - 新
]

# Easy组：使用不同深度的蓝色系
easy_colors = ['#1565C0', '#0277BD', '#00838F']

# 按组分类
hard_datasets = df[df["group"] == "hard"]["dataset"].tolist()
easy_datasets = df[df["group"] == "easy"]["dataset"].tolist()

# ====== 绘制hard组（星号，更大更显眼）======
for i, dataset in enumerate(hard_datasets):
    subset = df[df["dataset"] == dataset]
    ax.scatter(
        subset["dci"], subset["efi"],
        c=hard_colors[i % len(hard_colors)], 
        s=700,  # 增大尺寸
        edgecolor='white', 
        linewidth=2.0,  # 增加边缘宽度
        label=dataset, 
        alpha=0.95, 
        marker="*",
        zorder=3  # 确保在网格上方
    )

# ====== 绘制easy组（圆圈）======
for i, dataset in enumerate(easy_datasets):
    subset = df[df["dataset"] == dataset]
    ax.scatter(
        subset["dci"], subset["efi"],
        c=easy_colors[i % len(easy_colors)], 
        s=280,  # 增大尺寸
        edgecolor='white', 
        linewidth=2.0,
        label=dataset, 
        alpha=0.95, 
        marker="o",
        zorder=3
    )

# ====== 图表美化 ======
ax.set_title("Hardness of Different Datasets", fontsize=18, weight='bold', pad=20)
ax.set_xlabel("DCI", fontsize=18, weight='semibold')
ax.set_ylabel("EFI", fontsize=18, weight='semibold')

# 设置坐标轴范围，留出适当边距
ax.set_xlim(-0.2, 4.2)
ax.set_ylim(0.95, 2.05)

# 调整刻度字体大小
ax.tick_params(axis='both', labelsize=14)

# ====== 优化图例 ======
# 创建两列图例，分组显示
handles, labels = ax.get_legend_handles_labels()

# 重新排序：hard组在前，easy组在后
hard_indices = [i for i, label in enumerate(labels) if label in hard_datasets]
easy_indices = [i for i, label in enumerate(labels) if label in easy_datasets]
ordered_indices = hard_indices + easy_indices
ordered_handles = [handles[i] for i in ordered_indices]
ordered_labels = [labels[i] for i in ordered_indices]

# 创建图例
legend = ax.legend(
    ordered_handles, ordered_labels,
    title="Dataset", 
    loc='upper right',
    ncol=2,  # 两列显示
    fontsize=10,
    title_fontsize=12,
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='gray',
    columnspacing=1.0,
    handletextpad=0.5,
    markerscale=0.8
)

# 加粗图例标题
legend.get_title().set_weight('bold')

# ====== 添加网格样式 ======
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_axisbelow(True)

# 设置背景色
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# ====== 添加轻微的边框 ======
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.2)

plt.tight_layout()

# 保存 & 显示
plt.savefig("figs/MitoHard_complexity.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig("figs/MitoHard_complexity.pdf", bbox_inches='tight', facecolor='white')  # 额外保存PDF版本
plt.show()