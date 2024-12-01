import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 假设你的DataFrame叫做df，其中包含你的数据

# 设置绘图风格
sns.set(style="whitegrid")

# 创建一个图形对象和子图
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 15))
axes = axes.flatten()

# 获取所有列名（不包括id和洪水概率）
columns = [col for col in df.columns if col not in ['id', '洪水概率']]

# 绘制每个变量的箱线图
for ax, col in zip(axes, columns):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)

# 如果有剩余的子图，关闭它们
if len(columns) < len(axes):
    for unused_ax in axes[len(columns):]:
        fig.delaxes(unused_ax)

plt.tight_layout()
plt.show()