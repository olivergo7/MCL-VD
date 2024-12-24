import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# 1. 加载数据集
data = pd.read_csv('devign_processed.csv')

# 2. 加载 CodeT5 模型和分词器
model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. 设置批量生成参数
batch_size = 10  # 每批处理的代码数量，可根据资源调整
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 4. 函数：生成注释（改进后的 prompt 和生成设置）
def generate_comment_for_code(code_snippet):
    input_text = f"Write a detailed and clear comment explaining the purpose and functionality of the following code:\n\n{code_snippet}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,  # 增加 beam search 的宽度
            early_stopping=True,  # 早停以防止生成冗长内容
            no_repeat_ngram_size=2  # 防止生成重复片段
        )

    comment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return comment

# 5. 为缺少注释的代码生成注释
new_comments = []
total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

for idx in tqdm(range(0, len(data), batch_size), desc="Generating comments", total=total_batches):
    batch = data.iloc[idx: idx + batch_size]
    generated_comments = []

    for _, row in batch.iterrows():
        if pd.isnull(row['comments']) or row['comments'].strip() == "":
            # 生成注释
            comment = generate_comment_for_code(row['func'])
            generated_comments.append(comment)
        else:
            # 保留原有注释
            generated_comments.append(row['comments'])

    # 更新批量生成的注释
    new_comments.extend(generated_comments)

    # 将进度逐步保存至新文件
    data.loc[idx: idx + batch_size - 1, 'comments'] = generated_comments
    data.iloc[idx: idx + batch_size].to_csv("devign_processed_with_comments.csv", mode='a', index=False, header=idx == 0)

# 6. 最终完整保存至 CSV 文件
data.to_csv("devign_processed_with_comments_complete.csv", index=False)
print("注释生成完毕，已保存至 devign_processed_with_comments_complete.csv")
