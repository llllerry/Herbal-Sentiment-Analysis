import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import os

# ==========================================
# 1. 准备环境与加载本地模型
# ==========================================
# 指向我们刚刚保存的本地文件夹
model_dir = "./saved_bert_model"

if not os.path.exists(model_dir):
    print(f"❌ 找不到模型文件夹 {model_dir}，请确认路径！")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在 {device} 上唤醒你的专属 BERT 模型，请稍等...")

# 【核心点】这次我们直接从本地文件夹加载，不再需要联网下载了！
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

# 将模型设置为评估模式（关闭 Dropout 等训练专属的随机机制）
model.eval()

# 标签的中文映射对照表
label_map = {
    0: '😡 负面 (Negative)', 
    1: '😐 中性 (Neutral)', 
    2: '😊 正面 (Positive)'
}

# ==========================================
# 2. 定义预测魔法函数
# ==========================================
def predict_sentiment(text):
    # 用你的 tokenizer 把输入的句子转成数字
    inputs = tokenizer(
        text, 
        return_tensors="pt",  # 返回 PyTorch 张量
        max_length=128, 
        truncation=True, 
        padding=True
    ).to(device)
    
    # 预测时不需要算梯度，直接 no_grad 极速运行
    with torch.no_grad():
        outputs = model(**inputs)
        # 拿到模型的原始打分
        logits = outputs.logits
        
        # 用 softmax 强行把打分变成百分比概率 (加起来等于 100%)
        probs = F.softmax(logits, dim=1).squeeze()
        
        # 找出概率最大的那个类别的序号 (0, 1, 或 2)
        pred_idx = torch.argmax(probs).item()
        
    return label_map[pred_idx], probs.cpu().numpy()

# ==========================================
# 3. 开启交互聊天界面
# ==========================================
print("\n" + "="*40)
print(" 🚀 中药评价情感分析机器人已启动！")
print("="*40)
print("您可以随便输入一句话来测试它的实力。")
print("输入字母 'q' 退出程序。\n")

while True:
    user_input = input("🗣️ 请输入一条评价: ")
    
    if user_input.strip().lower() == 'q':
        print("👋 拜拜！模型已休眠。")
        break
        
    if not user_input.strip():
        continue
        
    # 调用模型进行预测
    label, probs = predict_sentiment(user_input)
    
    print("-" * 30)
    print(f"👉 最终诊断: {label}")
    # 打印出模型内心的纠结程度（各个类别的概率）
    print(f"📊 信心指数: [负面: {probs[0]:.1%}] | [中性: {probs[1]:.1%}] | [正面: {probs[2]:.1%}]")
    print("-" * 30 + "\n")