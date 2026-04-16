import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os

# 强制让代码去访问国内的镜像站，避开网络阻拦


# ==========================================
# 1. 数据读取与预处理
# ==========================================
def load_and_preprocess_data(csv_path):
    print(f"开始从 {csv_path} 读取数据...")
    df = pd.read_csv(csv_path)
    
    # 只保留需要的列，并去除可能存在的空值行
    df = df[['review_text', 'sentiment_label']].dropna()
    
    # 将英文标签映射为模型认识的数字
    label_mapping = {
        'negative': 0, # 负面
        'neutral': 1,  # 中性
        'positive': 2  # 正面
    }
    df['label'] = df['sentiment_label'].map(label_mapping)
    print(f"数据清洗完成，共保留 {len(df)} 条有效数据。\n")
    return df

# ==========================================
# 2. 构建词表与文字转数字 (Tokenization)
# ==========================================
def build_vocab(texts, max_size=5000):
    print("正在翻阅全体评论，建立汉字字典...")
    counter = Counter()
    for text in texts:
        counter.update(list(str(text))) # 把句子拆成单个字统计
    
    # 0给填充(PAD)，1给生僻字(UNK)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    print(f"字典建立完成，收录了 {len(vocab)} 个常用字。\n")
    return vocab

def text_to_sequence(texts, vocab, max_len=100):
    print("正在将汉字翻译为数字序列，准备喂给模型...")
    sequences = []
    for text in texts:
        # 查字典转换
        seq = [vocab.get(char, vocab['<UNK>']) for char in list(str(text))]
        
        # 统一长度：长的切掉，短的补0
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
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        # 注意力权重层
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        # 最终的分类全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x) 
        lstm_out, _ = self.lstm(embedded)
        
        # 计算注意力得分
        scores = self.attention_weights(lstm_out) 
        alpha = F.softmax(scores, dim=1) 
        
        # 融合成代表整句话的向量
        alpha = alpha.transpose(1, 2)                       
        context_vector = torch.bmm(alpha, lstm_out)         
        context_vector = context_vector.squeeze(1)          
        
        # 输出 3 个类别的得分
        out = self.fc(context_vector) 
        return out

# ==========================================
# 4. 主程序：流水线组装与训练启动
# ==========================================
if __name__ == '__main__':
    # 1. 确认文件路径 (根据你上传的文件夹结构)
    csv_file_path = "data/herbal_sentiment_train.csv"
    
    if not os.path.exists(csv_file_path):
        print(f"哎呀，找不到文件：{csv_file_path}，请检查路径哦！")
        exit()

    # 2. 读取数据
    df = load_and_preprocess_data(csv_file_path)
    
    # 【小贴士】为了让你马上看到训练效果，这里我们先取前 5000 条数据试跑。
    # 等代码跑通了，你可以把这行注释掉，用全部数据训练。
    df = df.head(5000) 
    print("注意：当前为快速验证模式，仅使用前 5000 条数据。\n")
    
    texts = df['review_text'].tolist()
    labels = df['label'].tolist()
    
    # 3. 数字化处理
    vocab = build_vocab(texts)
    X = text_to_sequence(texts, vocab, max_len=100)
    Y = torch.tensor(labels, dtype=torch.long)
    
    # 4. 打包给 PyTorch
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 5. 准备模型和计算设备
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
    
    # 6. 正式开始训练
    num_epochs = 5
    print("\n--- 🚀 开始训练模型 ---")
    
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0
        
        for batch_idx, (batch_X, batch_Y) in enumerate(dataloader):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()      # 清空之前的梯度
            outputs = model(batch_X)   # 预测
            loss = criterion(outputs, batch_Y) # 算误差
            loss.backward()            # 误差反向传播
            optimizer.step()           # 更新参数
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"✅ 第 {epoch+1}/{num_epochs} 轮结束 | 平均误差 (Loss): {avg_loss:.4f}")

    print("\n🎉 太棒了！Bi-LSTM 模型的训练主流程已经全部跑通！")
