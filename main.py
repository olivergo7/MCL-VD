import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
# Increase recursion limit to avoid deep recursion issues
sys.setrecursionlimit(10000)

# Load the processed Reveal dataset (modify the path as necessary)
file_path = 'reveal_processed.csv'
reveal_data = pd.read_csv(file_path)


# Custom Dataset class for DataLoader
class CodeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'code_text': str(item['code']),  # 确保是字符串
            'ast': str(item['ast']),  # 确保是字符串
            'comments': str(item['comments']),  # 确保是字符串
            'label': item['label']
        }


# Split the dataset into training, validation, and test sets (80%, 10%, 10%)
train_data, test_data = train_test_split(reveal_data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Create datasets and dataloaders
train_dataset = CodeDataset(train_data)
valid_dataset = CodeDataset(valid_data)
test_dataset = CodeDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GraphCodeBERT as a classification model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=2)  # 假设是二分类任务
model = model.to(device)

# 确保日志和模型保存的目录存在
save_directory = 'model/reveal'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 设置日志文件
log_file = os.path.join(save_directory, 'training_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')


# Function to encode text using GraphCodeBERT, and ensure tensors are on the same device
def encode_text(text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        device)  # Move inputs to the correct device
    outputs = model.roberta(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(
        1)  # Use the [CLS] token representation and remove extra dimension


# Contrastive loss function (NT-Xent Loss)
def contrastive_loss(features1, features2, temperature=0.5):
    # Ensure the input features have the correct shape (batch_size, hidden_size)
    features1 = features1.squeeze(1)  # Remove extra dimensions if necessary
    features2 = features2.squeeze(1)

    # Normalize the feature vectors
    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features1, features2.T)  # Shape: [batch_size, batch_size]

    # Create labels (diagonal elements are positive samples)
    labels = torch.arange(features1.shape[0]).to(features1.device)

    # Compute contrastive loss
    loss_i = F.cross_entropy(similarity_matrix / temperature, labels)
    loss_j = F.cross_entropy(similarity_matrix.T / temperature, labels)

    return (loss_i + loss_j) / 2


# Validation and testing function to compute evaluation metrics
def evaluate_model(model, data_loader):
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            code_texts = batch['code_text']
            labels = batch['label'].to(device)

            # Skip samples without comments
            valid_indices = [i for i, comment in enumerate(batch['comments']) if
                             isinstance(comment, str) and comment.strip() != ""]
            if len(valid_indices) == 0:
                continue

            # Filter valid samples
            code_texts = [code_texts[i] for i in valid_indices]
            labels = labels[valid_indices]

            # Encode code text
            code_text_reps = torch.stack([encode_text(text, device) for text in code_texts]).to(device)

            # Classification prediction
            classification_logits = model.classifier(code_text_reps)
            preds = torch.argmax(classification_logits, dim=1)

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return accuracy, precision, recall, f1


# 初始化数据存储列表
train_losses = []
valid_accuracies = []
valid_precisions = []
valid_recalls = []
valid_f1_scores = []

# 记录日志
logging.info("Training started.")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 10
# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        code_texts = batch['code_text']
        ast_texts = batch['ast']
        comments = batch['comments']
        labels = batch['label'].to(device)

        valid_indices = [i for i, comment in enumerate(comments) if isinstance(comment, str) and comment.strip() != ""]
        if len(valid_indices) == 0:
            continue

        # Filter valid samples
        code_texts = [code_texts[i] for i in valid_indices]
        ast_texts = [ast_texts[i] for i in valid_indices]
        comments = [comments[i] for i in valid_indices]
        labels = labels[valid_indices]

        # Encode code text, AST, and comments, and move them to GPU (if available)
        code_text_reps = torch.stack([encode_text(text, device) for text in code_texts]).to(device)
        ast_reps = torch.stack([encode_text(text, device) for text in ast_texts]).to(device)
        comment_reps = torch.stack([encode_text(text, device) for text in comments]).to(device)

        # Compute classification loss
        classification_logits = model.classifier(code_text_reps)
        classification_loss = F.cross_entropy(classification_logits, labels)

        # Compute contrastive loss
        loss_code_ast = contrastive_loss(code_text_reps, ast_reps)
        loss_code_comment = contrastive_loss(code_text_reps, comment_reps)
        loss_ast_comment = contrastive_loss(ast_reps, comment_reps)

        contrastive_loss_total = (loss_code_ast + loss_code_comment + loss_ast_comment) / 3

        # Total loss
        total_loss = classification_loss + contrastive_loss_total

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss += total_loss.item()

    train_losses.append(total_loss / len(train_loader))

    # 验证集上的评估
    accuracy, precision, recall, f1 = evaluate_model(model, valid_loader)
    valid_accuracies.append(accuracy)
    valid_precisions.append(precision)
    valid_recalls.append(recall)
    valid_f1_scores.append(f1)

    # 打印训练进度
    logging.info(
        f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# 保存模型
save_path = os.path.join(save_directory, 'graphcodebert_model.pth')
torch.save(model.state_dict(), save_path)
logging.info(f"Model saved to {save_path}")
print(f"Model saved to {save_path}")

# 加载并测试模型
model.load_state_dict(torch.load(save_path))
model.eval()

# 测试集上的评估
accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
logging.info(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

