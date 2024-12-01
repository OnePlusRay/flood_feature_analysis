import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
file_path = '../data/train_20000.csv'
df = pd.read_csv(file_path)

# 假设 '洪水概率' 是目标变量
X = df.drop(columns=['洪水概率'])
y = df['洪水概率']

# 2. 数据预处理
# 检查和处理缺失值（可以选择删除或填充缺失值）
X.fillna(X.mean(), inplace=True)  # 用均值填充缺失值

# 对于类别特征进行独热编码（如果有）
X = pd.get_dummies(X)

# 特征标准化（可选，基于模型需要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 构建和训练 XGBoost 模型
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5. 预测和评估模型性能
y_pred = model.predict(X_test)

# 计算性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("均方误差:", mse)
print("R^2 分数:", r2)

# 重要性分析
importance = model.feature_importances_
importance_df = pd.DataFrame(importance, index=X.columns, columns=['特征重要性']).sort_values(by='特征重要性', ascending=False)

# 绘制特征重要性图
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
importance_df.plot(kind='bar')
plt.title('特征重要性')
plt.ylabel('重要性')
plt.xlabel('特征')
plt.show()
