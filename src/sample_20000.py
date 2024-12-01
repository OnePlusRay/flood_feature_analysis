import pandas as pd
import random
import chardet

# 定义文件路径
file_path = '../data/train.csv'

# 读取原始数据，直接指定编码格式
df = pd.read_csv(file_path, encoding='utf-8')

# 检测文件编码
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']

# 设置随机种子
random.seed(42)

# 随机抽取 2 万条数据
sample_size = 20000
sample_indices = random.sample(range(len(df)), sample_size)  # 从索引范围中随机选择样本索引
df_sample = df.iloc[sample_indices]  # 根据选择的索引筛选数据

# 保存抽取的数据
output_path = '../data/train_20000.csv'
df_sample.to_csv(output_path, index=False, encoding='utf-8')  # 保存为 utf-8 编码

print(f"已成功从 {file_path} 中随机抽取 2 万条数据，并保存为 {output_path}")
