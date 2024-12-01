import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

# 从Excel文件中读取数据
df = pd.read_excel('../data/train_20000.xlsx')  # 修改为Excel读取

# 显示数据的前几行以确认读取成功
print(df.head())

# 分离自变量（X）和因变量（y）
X = df.drop(columns=['id', '洪水概率'])  # 假设 'id' 列不参与回归
y = df['洪水概率']

# 划分数据集（80%训练集，20%测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()
ridge_model = Ridge(alpha=1.0)

# 训练模型
model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)
y_ridge_pred = ridge_model.predict(X_test)

# 计算决定系数（R^2）
r2 = r2_score(y_test, y_pred)
r2_ridge = r2_score(y_test, y_ridge_pred)
print(f'线性回归决定系数 (R^2): {r2}')
print(f'岭回归决定系数 (R^2): {r2_ridge}')

# # 打印模型参数
# print("模型参数（回归系数）:")
# for feature, coef in zip(X.columns, model.coef_):
#     print(f'{feature}: {coef:.4f}')
    
# print(f'截距: {model.intercept_:.4f}')


