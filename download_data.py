from datasets import load_dataset
import pandas as pd
import os

print("正在努力从 Hugging Face 获取数据，请稍等...")

# 1. 自动下载数据集
# 这里直接填入你发的数据集名字
dataset = load_dataset("OpenModels/Chinese-Herbal-Medicine-Sentiment")

# Hugging Face 的数据通常会分块（比如 train, test），我们先把核心的 train 部分拿出来
# 并把它转换成我们最熟悉的 Pandas 表格格式
df = dataset['train'].to_pandas()

print(f"数据获取成功！一共拿到了 {len(df)} 行数据。")

# 2. 准备保存到本地
# 我们可以建一个文件夹专门放数据，显得清爽一点
save_dir = "./data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 完整的文件保存路径
save_path = f"{save_dir}/herbal_sentiment_train.csv"

# 3. 写入文件
# index=False 代表不保存最左边的 0,1,2 序号
# encoding='utf-8-sig' 是一个小技巧，能保证用 Excel 打开带有中文的 CSV 时不会乱码
df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"太棒了！数据已经成功保存到本地电脑：{save_path}")