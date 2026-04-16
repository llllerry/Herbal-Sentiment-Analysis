import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

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
# 2. 构建词表与文字转数字
# ==========================================
def build_vocab(texts, max_size=5000):
    print("正在建立汉字字典...")
    counter = Counter()
    for text in texts:
        counter.update(list(str(text)))
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

def text_to_sequence(texts, vocab, max_len=100):
    sequences = []
    for text in texts:
        seq = [vocab.get(char, vocab['<UNK>']) for char in list(str(text))]
        if len(seq) < max_len:
            seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.long)

# ==========================================
# 3. 定义 Bi-LSTM + Attention 模型架构
# ==========================================
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                            batch_first=True, bidirectional=True)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x) 
        lstm_out, _ = self.lstm(embedded)
        
        scores = self.attention_weights(lstm_out) 
        alpha = F.softmax(scores, dim=1) 
        
        alpha = alpha.transpose(1, 2)                       
        context_vector = torch.bmm(alpha, lstm_out)         
        context_vector = context_vector.squeeze(1)          
        
        out = self.fc(context_vector) 
        return out

# ==========================================
# 4. 考试评估函数
# ==========================================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

# ==========================================
# 5. 主程序：全量开跑
# ==========================================
if __name__ == '__main__':
    csv_file_path = "data/herbal_sentiment_train.csv"
    df = load_and_preprocess_data(csv_file_path)
    
    # 【改动点】按照 9:1 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"✅ 划分完成！训练集: {len(train_df)} 条，测试集(考试题): {len(test_df)} 条。\n")
    
    # 注意：字典一定要只用训练集来建，不能让模型提前“偷看”测试集里的字
    vocab = build_vocab(train_df['review_text'].tolist())
    
    # 数字化处理
    print("正在将文字转为数字矩阵...")
    X_train = text_to_sequence(train_df['review_text'].tolist(), vocab)
    Y_train = torch.tensor(train_df['label'].tolist(), dtype=torch.long)
    
    X_test = text_to_sequence(test_df['review_text'].tolist(), vocab)
    Y_test = torch.tensor(test_df['label'].tolist(), dtype=torch.long)
    
    # 打包给 PyTorch
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    
    # Bi-LSTM 比较轻量，batch_size 可以开到 128 或 256 提速
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---> 当前使用的计算设备是: {device} <---")
    
    model = BiLSTMAttention(
        vocab_size=len(vocab), 
        embedding_dim=128,   
        hidden_dim=128,      
        num_classes=3        
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 3
    print("\n--- 🚀 开始全量 Bi-LSTM 训练 ---")
    
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0
        
        # 使用 tqdm 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_X, batch_Y in progress_bar:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()      
            outputs = model(batch_X)   
            loss = criterion(outputs, batch_Y) 
            loss.backward()            
            optimizer.step()           
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        
        # 评估环节
        print(f"\n⏳ 第 {epoch+1} 轮训练完成，正在测试集上评估...")
        val_acc, val_f1 = evaluate(model, test_loader, device)
        print(f"📊 Bi-LSTM 成绩单 -> 准确率: {val_acc*100:.2f}% | F1-Score: {val_f1*100:.2f}%\n")

    print("🎉 恭喜！Bi-LSTM 全量数据训练与评估完成！")