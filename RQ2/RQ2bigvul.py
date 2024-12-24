# 导入必要的库
import os
import logging
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 加载数据集
file_path = 'bigvul_modified.csv'
bigvul_data = pd.read_csv(file_path)


# 自定义 Dataset 类
class CodeDataset(Dataset):
    def __init__(self, data, mode='all'):
        self.data = data.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        code_text = str(item['func'])
        ast = str(item['ast'])
        comments = str(item['comments'])
        label = item['target']

        if not code_text.strip():
            code_text = "empty"
        if not ast.strip():
            ast = "empty"
        if not comments.strip():
            comments = "empty"

        # 根据模式选择输入内容
        if self.mode == 'code':
            return {'input': code_text, 'label': label}
        elif self.mode == 'ast':
            return {'input': ast, 'label': label}
        elif self.mode == 'comments':
            return {'input': comments, 'label': label}
        elif self.mode == 'code+comments':
            return {'input': code_text + " " + comments, 'label': label}
        elif self.mode == 'code+ast':
            return {'input': code_text + " " + ast, 'label': label}
        elif self.mode == 'comments+ast':
            return {'input': comments + " " + ast, 'label': label}
        elif self.mode == 'all':
            return {'input': code_text + " " + ast + " " + comments, 'label': label}
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


# 数据集划分（分层采样）
train_data, test_data = train_test_split(
    bigvul_data, test_size=0.2, random_state=seed, stratify=bigvul_data['target']
)
valid_data, test_data = train_test_split(
    test_data, test_size=0.5, random_state=seed, stratify=test_data['target']
)

# 创建数据集和数据加载器
mode = 'all'  # 选择输入模式，可以是 'code', 'ast', 'comments', 'code+comments', 'code+ast', 'comments+ast', 'all'
train_dataset = CodeDataset(train_data, mode=mode)
valid_dataset = CodeDataset(valid_data, mode=mode)
test_dataset = CodeDataset(test_data, mode=mode)

batch_size = 8  # 根据显存情况调整
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 提取验证集和测试集的标签
valid_labels = [data['label'] for data in valid_dataset]
test_labels = [data['label'] for data in test_dataset]

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 GraphCodeBERT 模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
base_model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=2)
base_model = base_model.to(device)

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,  # 增大 r 值
    lora_alpha=32,  # 增大 lora_alpha
    lora_dropout=0.1,
)

# 将 LoRA 应用于模型
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 解冻最后三层 Transformer 层和分类器
for name, param in model.named_parameters():
    if any(f"layer.{i}" in name for i in [9, 10, 11]) or "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 确保保存目录存在
save_directory = 'model/bigvul/rq2'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 设置日志记录
log_file = os.path.join(save_directory, 'training_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# 初始化 TensorBoard
writer = SummaryWriter(log_dir=save_directory)

# 优化器和线性学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
epochs = 50
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

# 混合精度训练初始化
scaler = GradScaler()

# 加权损失函数
weights = compute_class_weight('balanced', classes=[0, 1], y=train_data['target'].values)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)


# 对比损失函数
def contrastive_loss(features1, features2, temperature=0.5):
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)
    logits = torch.matmul(features1, features2.T) / temperature
    labels = torch.arange(features1.size(0)).to(features1.device)
    return F.cross_entropy(logits, labels)


# 添加函数：计算最佳阈值
def find_best_threshold(labels, probs, optimize='f1'):
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    if optimize == 'f1':
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_threshold = thresholds[np.argmax(f1_scores)]
    elif optimize == 'precision':
        best_threshold = thresholds[np.argmax(precision)]
    elif optimize == 'recall':
        best_threshold = thresholds[np.argmax(recall)]
    else:
        raise ValueError("Invalid 'optimize' parameter. Choose from 'f1', 'precision', 'recall'.")

    return best_threshold


# 评估函数
def evaluate_model(model, data_loader, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = tokenizer(batch['input'], return_tensors="pt", truncation=True, padding=True, max_length=256).to(
                device)
            labels = batch['label'].to(device)

            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = (probs > threshold).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    average_loss = total_loss / len(data_loader)

    return accuracy, precision, recall, f1, average_loss, all_probs


# 训练循环
best_f1 = 0
early_stop_count = 0
early_stop_patience = 10

logging.info("Training started.")
accumulation_steps = 4  # 梯度累积步数

for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

    for step, batch in enumerate(progress_bar):
        inputs = tokenizer(batch['input'], return_tensors="pt", truncation=True, padding=True, max_length=256).to(
            device)
        labels = batch['label'].to(device)

        with autocast():
            logits = model(**inputs).logits
            classification_loss = loss_fn(logits, labels) / accumulation_steps
            total_batch_loss = classification_loss

        scaler.scale(total_batch_loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += total_batch_loss.item()
        progress_bar.set_postfix(loss=total_batch_loss.item())

    average_loss = total_loss / len(train_loader)
    # 在验证集上评估
    accuracy, precision, recall, f1, valid_loss, valid_probs = evaluate_model(model, valid_loader)
    best_threshold = find_best_threshold(valid_labels, valid_probs, optimize='f1')
    valid_preds = (np.array(valid_probs) > best_threshold).astype(int)
    precision = precision_score(valid_labels, valid_preds, zero_division=0)
    recall = recall_score(valid_labels, valid_preds, zero_division=0)
    f1 = f1_score(valid_labels, valid_preds, zero_division=0)

    log_message = (f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}, "
                   f"Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}, "
                   f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Threshold: {best_threshold:.4f}")
    logging.info(log_message)
    print(log_message)

    # 早停机制
    if f1 > best_f1:
        best_f1 = f1
        early_stop_count = 0
        save_path = os.path.join(save_directory, 'best_graphcodebert_model.pth')
        torch.save(model.state_dict(), save_path)
    else:
        early_stop_count += 1
        if early_stop_count >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

print("Training complete.")

model.load_state_dict(torch.load(os.path.join(save_directory, 'best_graphcodebert_model.pth')))

# 测试评估
accuracy, precision, recall, f1, test_loss, test_probs = evaluate_model(model, test_loader)
best_threshold = find_best_threshold(test_labels, test_probs)
test_preds = (np.array(test_probs) > best_threshold).astype(int)
test_precision = precision_score(test_labels, test_preds, zero_division=0)
test_recall = recall_score(test_labels, test_preds, zero_division=0)
test_f1 = f1_score(test_labels, test_preds, zero_division=0)

test_message = (f"Test Results - Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, "
                f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, Threshold: {best_threshold:.4f}")
print(test_message)
logging.info(test_message)

writer.close()
