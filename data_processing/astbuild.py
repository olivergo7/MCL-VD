import pandas as pd
import re
from tree_sitter import Language, Parser

# 加载 Reveal 数据集
file_path = 'data/reveal.csv'
reveal_data = pd.read_csv(file_path)

# 编译 tree-sitter 以支持 C 语言，修改路径为适合你环境的路径
Language.build_library(
    'build/my-languages.so',
    [
        'tree-sitter-c',  # 支持 C 语言
        # 如果需要支持其他语言，可以在这里添加
        # 如 'tree-sitter-python', 'tree-sitter-javascript', 等
    ]
)

