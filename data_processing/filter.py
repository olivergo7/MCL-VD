import pandas as pd
import re

# 读取数据集
df = pd.read_csv('bigvul_processed_filtered.csv')# 修改为你文件的路径

# 定义删除注释的函数
def remove_comments(code):
    # 使用正则表达式删除 # 之后的内容
    return re.sub(r'#.*', '', code).strip()

# 应用函数到 'func' 列
df['func'] = df['func'].apply(remove_comments)

# 如果你需要保存修改后的数据集
df.to_csv('bigvul_modified.csv', index=False)
