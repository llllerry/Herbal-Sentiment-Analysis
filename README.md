# 基于深度学习的中药领域用户评价情感分类实践
## 📊 数据集说明与声明

本项目使用的核心训练数据 `herbal_sentiment_train.csv` 包含逾 21 万条真实的中药垂直领域评价。

* **数据来源**：[下载自 Hugging Face 某开源数据集OpenModels/Chinese-Herbal-Medicine-Sentiment]。
* **数据格式**：经过脱敏与预处理，数据仅保留了 `review_text` (评价文本) 与 `sentiment_label` (情感倾向) 两个核心字段。
* **⚠️ 免责声明 (Disclaimer)**：
  本项目所涉及的数据集、预训练模型权重及相关代码，仅供**学术研究、技术交流与个人学习**使用，严禁用于任何商业用途。数据来源于公开网络，若相关数据或文本侵犯了您的合法权益，请提请 Issue 或联系作者，我们将第一时间进行删除处理。
## 📖 项目背景
本项目旨在处理中药电商场景下的真实长短文本评价数据（逾 21 万条）。由于真实的业务数据呈现极度的正负样本不平衡分布（Positive 样本占主导），常规模型极易产生预测偏移并陷入**过拟合**。本项目通过对比不同深度的网络架构，探索模型在长尾非平衡数据下的特征提取能力与泛化表现。

## 📦 模型下载与调用

本项目的完整 BERT 模型权重（基于 `bert-base-chinese` 微调）已开源至 Hugging Face，你可以直接通过 `transformers` 库一键调用：

**Hugging Face 仓库地址**: [https://huggingface.co/1hugh/Herbal-Sentiment-BERT](https://huggingface.co/1hugh/Herbal-Sentiment-BERT)

**Python 快速调用代码**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("1hugh/Herbal-Sentiment-BERT")
model = AutoModelForSequenceClassification.from_pretrained("1hugh/Herbal-Sentiment-BERT")

# 准备测试文本
text = "这当归发霉了，味道极差！"
inputs = tokenizer(text, return_tensors="pt")

# 进行情感预测
outputs = model(**inputs)
print(outputs.logits.argmax().item()) 
# 输出结果对应: 0:负面, 1:中立, 2:正面

## 🛠️ 技术路线与对比实验
本项目未直接依赖高度封装的 API，而是通过对比两套经典架构，验证不同**归纳偏置（Inductive Bias）**对文本情感分析的影响：

1. **基准模型 (Baseline)：基于 PyTorch 手写的 Bi-LSTM + Attention**
   * **实现细节**：从零构建字符级词表（Vocab），独立处理序列截断与 `<PAD>` 填充。利用双向 LSTM 的时序记忆特性捕捉上下文依赖，并结合 Attention 机制动态赋予核心情感词高额权重。
   * **实验目的**：验证序列模型在中文基础情感分类上的有效性，确立非平衡数据下的评估基准。

2. **进阶方案：基于 BERT 的微调调优 (Fine-tuning)**
   * **实现细节**：引入 `bert-base-chinese` 预训练模型，利用 Transformer 强大的多头自注意力机制（Self-Attention）深度挖掘中药专有名词与情感词汇的关联语义。
   * **实验成果**：在两万余条未见过的测试集上，取得了 **89.36% 的准确率（Accuracy）**和 **77.08% 的宏平均 F1-Score**，精准突破了少数类（负面/中立评价）的识别瓶颈。

## 📂 项目结构
```text
├── Bi-LSTM.py        # 基于 PyTorch 的基线模型快速验证脚本（抽取小样本测试代码连通性）
├── Bi-LSTM_full.py   # Bi-LSTM 全量数据（21万条）训练与评估完整脚本
├── BERT.py           # 引入 Hugging Face 生态的 BERT 微调全量训练脚本
├── predict.py        # 封装好的交互式端到端推理脚本
├── download_data.py  # Hugging Face 数据集快速拉取脚本
└── README.md         # 项目说明文档
