import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 从CSV文件中读取数据
df = pd.read_csv('/Users/10240013/Documents/学校资料/洪水问题一/train_20000.csv')

# 显示数据的前几行以确认读取成功
print(df.head())

# 分离自变量（X）和因变量（y）
X = df.drop(columns=['id', '洪水概率'])  # 假设 'id' 列不参与回归
y = df['洪水概率']

# 划分数据集（80%训练集，20%测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算决定系数（R^2）
r2 = r2_score(y_test, y_pred)
print(f'决定系数 (R^2): {r2}')
