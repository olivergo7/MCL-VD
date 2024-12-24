import pandas as pd
import openai
import time

# 设置 OpenAI API 基础 URL 和 API 密钥
openai.api_base = 'https://api.closeai-proxy.xyz/v1'
openai.api_key = 'your_api_key'# 修改为你的api

# 加载数据集
data_path = "bigvul.csv"# 修改为你文件的路径
data = pd.read_csv(data_path)
print(f"加载数据集完成，行数: {len(data)}")

# 使用 GPT-4o-mini 生成注释
def generate_comment_for_code(code_snippet):
    prompt = f"""Please provide a clear and concise English comment for the following C function.
    Match the style and detail of existing comments in this dataset.
    The comment should include:
    - A brief function summary
    - Descriptions of input parameters
    - Expected output or return value
    - Notes on any exceptions or potential issues, if applicable

    Code:
    {code_snippet}

    Only return the comment content."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are an experienced C developer skilled in writing concise and clear comments, matching existing dataset styles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300,
            n=1,
            stop=None,
        )
        comment = response['choices'][0]['message']['content'].strip()
        print(f"Generated comment: {comment}")  # 打印生成的注释
        return comment
    except Exception as e:
        print(f"Error generating comment: {e}")
        return ""

# 为缺少注释的代码片段生成注释并更新数据集
save_interval = 200  # 每100条保存一次
counter = 0  # 初始化计数器

for index, row in data.iterrows():
    if pd.isnull(row['comments']) or row['comments'].strip() == "":
        # 生成注释
        print(f"Processing code snippet {index + 1}/{len(data)}")
        comment = generate_comment_for_code(row['func'])
        if comment:
            # 更新注释
            data.at[index, 'comments'] = comment
            print(f"Updated comment for function at index {index}")

            counter += 1  # 更新计数器

            # 如果计数达到保存间隔，保存文件并重置计数器
            if counter >= save_interval:
                data.to_csv(data_path, index=False)
                print(f"Data saved to {data_path} after {save_interval} updates")
                counter = 0  # 重置计数器

        # 控制请求频率
        time.sleep(1)

# 保存任何剩余的更新
if counter > 0:
    data.to_csv(data_path, index=False)
    print(f"Final data saved to {data_path}")

print("Comment generation complete.")
