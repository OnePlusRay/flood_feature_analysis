import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体以增强可读性
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Noto Sans CJK']  
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 加载数据
df = pd.read_excel('../data/train_20000.xlsx')

# 查看描述性统计
description = df.describe()
print(description)

# 检查缺失值
missing_values = df.isnull().sum()
print(missing_values)

# 可视化缺失值
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('缺失值热图')
plt.show()

# 设置画布大小
plt.figure(figsize=(15, 10))

# 绘制直方图
for i, column in enumerate(df.columns[1:21], start=1):  # 跳过 'id' 列
    plt.subplot(4, 5, i)  # 根据列数决定子图结构
    sns.histplot(df[column], bins=10, kde=True, color='lightcoral')
    plt.title(column)
    plt.xlabel('分数')
    plt.ylabel('频数')
    
plt.tight_layout()
plt.savefig('../results/visualization/features_histplot.png', dpi=300)
plt.show()

# 设置画布大小
plt.figure(figsize=(15, 10))

# 绘制箱线图（有明确的大小关系，可视化时箱线图比饼图更加合适））
for i, column in enumerate(df.columns[1:21], start=1):  # 跳过 'id' 列
    plt.subplot(4, 5, i)  # 创建子图
    sns.boxplot(y=df[column], color='skyblue')
    plt.title(column)
    plt.ylabel('分数')
    
plt.tight_layout()
plt.savefig('../results/visualization/features_boxplot.png', dpi=300)
plt.show()

# 洪水概率直方图
plt.figure(figsize=(8, 5))
sns.histplot(df['洪水概率'], bins=10, kde=True, color='lightcoral')
plt.title('洪水概率分布')
plt.xlabel('洪水概率')
plt.ylabel('频数')
plt.grid()
plt.savefig('../results/visualization/flood_histplot.png', dpi=300)
plt.show()

# 洪水概率箱线图
plt.figure(figsize=(8, 5))
sns.boxplot(df['洪水概率'], color='skyblue')
plt.title('洪水概率分布')
plt.xlabel('洪水概率')
plt.ylabel('分数')
plt.grid()
plt.savefig('../results/visualization/flood_boxplot.png')
plt.show()



