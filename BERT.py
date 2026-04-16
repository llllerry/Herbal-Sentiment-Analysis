import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
from tqdm import tqdm  # 引入进度条库

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==========================================
# 1. 数据读取与预处理
# ==========================================
def load_and_preprocess_data(csv_path):
    print(f"开始读取全量数据...")
    df = pd.read_csv(csv_path)
    df = df[['review_text', 'sentiment_label']].dropna()
    
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['sentiment_label'].map(label_mapping)
    return df

# ==========================================
# 2. 专门为 BERT 准备的数据集格式
# ==========================================
class HerbalBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 3. 新增：模型评估函数 (在测试集上考试)
# ==========================================
def evaluate(model, dataloader, device):
    model.eval() # 切换到评估模式，模型在这个模式下不会更新参数（不会作弊）
    
    all_preds = []
    all_labels = []
    
    # 评估时不需要计算梯度，省显存又提速
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 拿到三个类别的得分，取分数最高的那个作为预测结果
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            # 把张量从显卡拉回 CPU，变成普通的 Python 列表，存起来
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 计算准确率和 F1 分数（由于数据不平衡，我们用 macro 模式计算综合 F1）
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return acc, f1

# ==========================================
# 4. 主程序：解开封印，全量开跑
# ==========================================
if __name__ == '__main__':
    csv_file_path = "data/herbal_sentiment_train.csv"
    df = load_and_preprocess_data(csv_file_path)
    
    # 【解除封印】：我们不再截取前 1000 条，直接使用全部数据！
    print(f"数据加载完毕，共 {len(df)} 条。准备进行训练集和测试集划分...")
    
    # 将全量数据按 9:1 的比例划分为训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"✅ 划分完成！训练集: {len(train_df)} 条，测试集(考试题): {len(test_df)} 条。\n")
    
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # --- 数据打包 ---
    train_dataset = HerbalBertDataset(train_df['review_text'].tolist(), train_df['label'].tolist(), tokenizer)
    test_dataset = HerbalBertDataset(test_df['review_text'].tolist(), test_df['label'].tolist(), tokenizer)
    
    # 注意：全量数据很大，如果你的 3050Ti 报 OOM (显存不足)，请把这里的 16 改成 8
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---> 当前使用的计算设备是: {device} <---")
    
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    num_epochs = 3
    print("\n--- 🚀 开始全量 BERT 微调训练 ---")
    
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0
        
        # 给 DataLoader 穿上 tqdm 的外衣，变成炫酷的进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 实时把 Loss 更新在进度条的后缀上
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        
        # 本轮训练结束，立刻用测试集“考试”
        print(f"\n⏳ 第 {epoch+1} 轮训练完成，平均误差: {avg_loss:.4f}。正在测试集上评估...")
        val_acc, val_f1 = evaluate(model, test_loader, device)
        
        print(f"📊 考试成绩单 -> 准确率 (Accuracy): {val_acc*100:.2f}% | F1-Score: {val_f1*100:.2f}%\n")

    print("🎉 恭喜！全量数据的微调和评估已全部完成！")
    
    # 提前为你写好保存模型的代码，留给步骤 C 使用
    save_dir = "./saved_bert_model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"💾 训练好的模型已成功保存至文件夹：{save_dir}")