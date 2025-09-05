import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ====== 从CSV文件读取数据 ======
csv_file_path = "/projects/weilab/liupeng/dataset/mito/complexity/MitoLE_efi_vs_contact_count_data.csv"
df = pd.read_csv(csv_file_path)

# ====== 设置 Seaborn 风格 ======
sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(10, 6))

# 为hard和easy组分别设置不同的色系
hard_colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(df[df["group"] == "hard"])))  # 暖色系（红色系）
easy_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(df[df["group"] == "easy"])))  # 冷色系（蓝色系）

# 绘制散点图，根据group使用不同色系
hard_datasets = df[df["group"] == "hard"]["dataset"].tolist()
easy_datasets = df[df["group"] == "easy"]["dataset"].tolist()

# 绘制hard组（星号，暖色系）
for i, dataset in enumerate(hard_datasets):
    subset = df[df["dataset"] == dataset]
    plt.scatter(
        subset["contact_count"], subset["efi"],
        c=[hard_colors[i]], s=150, edgecolor="black",
        label=dataset, alpha=0.8, marker="*"
    )

# 绘制easy组（圆圈，冷色系）
for i, dataset in enumerate(easy_datasets):
    subset = df[df["dataset"] == dataset]
    plt.scatter(
        subset["contact_count"], subset["efi"],
        c=[easy_colors[i]], s=150, edgecolor="black",
        label=dataset, alpha=0.8, marker="o"
    )

# 移除数据标签代码
# for _, row in df.iterrows():
#     plt.text(
#         row["contact_count"], row["efi"], row["dataset"],
#         fontsize=9, ha="right", va="bottom"
#     )

# ====== 图表美化 ======
plt.title("Hardness of Different Datasets", fontsize=14, weight="bold")
plt.xlabel("DCI", fontsize=16)
plt.ylabel("EFI", fontsize=16)

# 设置Y轴最大值为2
plt.ylim(top=2)

# 调整坐标轴刻度字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 将图例放到图里面的右上角
plt.legend(title="Dataset", loc='upper right')

plt.tight_layout()

# 保存 & 显示
plt.savefig("figs/scatter_plot_seaborn.png", dpi=600, bbox_inches='tight')
plt.show()
