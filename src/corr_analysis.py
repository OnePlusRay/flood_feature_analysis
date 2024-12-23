import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# 设置字体以增强可读性
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Noto Sans CJK']  
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义文件路径
file_path = '../data/train_20000.csv'

# 读取处理后的数据
df = pd.read_csv(file_path)
df = df.drop('id', axis=1)

# 查看数据基本信息
print(df.info())
print(df.head())

# 对所有列进行标准化操作
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 查看标准化后的数据
print(df_scaled.head())

# 计算斯皮尔曼相关性矩阵
correlation_matrix = df_scaled.corr(method='spearman')

# 提取与洪水概率的相关性
correlation_with_flood = correlation_matrix['洪水概率'].sort_values(ascending=False)

# 打印相关性
print(correlation_with_flood)

# 绘制相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=True, yticklabels=True)
plt.title('Spearman Correlation Matrix')
plt.xticks(rotation=45)  # 旋转 x 轴标签以避免重叠
plt.yticks(rotation=45)  # 旋转 y 轴标签以避免重叠
plt.savefig('../results/correlation_matrix.png', dpi=300)  # 保存为图片
plt.close()  # 关闭当前图形，以释放内存

# 绘制与洪水概率的相关性条形图
plt.figure(figsize=(10, 10))
correlation_with_flood.drop('洪水概率').plot(kind='bar')
plt.title('Spearman Correlation with Flood Probability')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.axhline(y=0, color='red', linestyle='--')
plt.xticks(rotation=45)  # 旋转 x 轴标签以避免重叠
plt.savefig('../results/correlation_with_flood.png', dpi=300)  # 保存为图片
plt.close()  # 关闭当前图形，以释放内存

# # 使用随机森林回归模型评估特征重要性
# X = df_scaled.drop(columns=['洪水概率'])
# y = df_scaled['洪水概率']

# # 训练随机森林回归模型
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X, y)

# # 获取特征重要性
# feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# # 打印特征重要性
# print(feature_importances)

# # 绘制特征重要性条形图
# plt.figure(figsize=(10, 10))
# feature_importances.plot(kind='bar')
# plt.title('Feature Importances from Random Forest Regressor')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.xticks(rotation=45)  # 旋转 x 轴标签以避免重叠
# plt.savefig('../results/feature_importances.png', dpi=300)  # 保存为图片
# plt.close()  # 关闭当前图形，以释放内存

# 分析相关性结果
highly_correlated_features = correlation_with_flood[abs(correlation_with_flood) > 0.3].index.tolist()
low_correlated_features = correlation_with_flood[abs(correlation_with_flood) < 0.1].index.tolist()

print("Highly Correlated Features:", highly_correlated_features)
print("Low Correlated Features:", low_correlated_features)

# 提出建议
suggestions = []
for feature in highly_correlated_features:
    if feature != '洪水概率':
        suggestions.append(f"{feature} 与洪水概率高度相关，建议重点关注和管理该指标。")

for feature in low_correlated_features:
    if feature != '洪水概率':
        suggestions.append(f"{feature} 与洪水概率相关性较低，可以适当减少对该指标的关注。")

print("Suggestions:")
for suggestion in suggestions:
    print(suggestion)

# 将相关性和特征重要性保存为 CSV 文件
correlation_with_flood.to_csv('../results/corr.csv')

# 获取与洪水概率相关性前6的特征
top_five_features = correlation_with_flood.index[1:]  # 排除 '洪水概率' 本身

# 绘制散点图
plt.figure(figsize=(15, 12))

for i, feature in enumerate(top_five_features):
    plt.subplot(4, 5, i + 1)  # 创建 3 行 2 列的子图
    sns.scatterplot(data=df, x=feature, y='洪水概率', alpha=0.7)
    plt.title(f'Scatter Plot: {feature} vs 洪水概率')
    plt.xlabel(feature)
    plt.ylabel('洪水概率')

plt.tight_layout()  # 自适应布局
plt.savefig('../results/top_six_features_scatter_plots.png', dpi=300)  # 保存为高清图片
plt.close()  # 关闭当前图形，以释放内存

print("分析结果已保存到文件中")
