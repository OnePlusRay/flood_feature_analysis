import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义文件路径
file_path = '/Users/10240013/Documents/学校资料/洪水问题一/train_20000.csv'

# 读取数据
df = pd.read_csv(file_path)

# 检查数据
print(df.info())
print(df.head())

# 特征列表（排除 'id' 和 '洪水概率' 列）
features = df.columns[1:-1]  # 取出所有特征名

# 设置绘图参数
plt.figure(figsize=(15, 20))

# 绘制散点图
for i, feature in enumerate(features):
    plt.subplot(4, 5, i + 1)  # 4行5列的子图
    sns.scatterplot(data=df, x=feature, y='洪水概率')
    plt.title(f'{feature} vs 洪水概率')
    plt.xlabel(feature)
    plt.ylabel('洪水概率')

plt.tight_layout()  # 自适应布局
plt.show()
