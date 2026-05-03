import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset
from torch.optim import AdamW
import numpy as np
import os
from datetime import datetime

# ================================================================
# 【TXT日志】初始化
# ================================================================
os.makedirs('./logs', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f'./logs/sentiment_{timestamp}.txt'

def log(msg):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

log('=' * 60)
log('  IMDb 情感分析 — BERT 迁移学习')
log(f'  开始时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log('=' * 60)
# ================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'distilbert-base-uncased'   # BERT 的轻量版，速度快约 2 倍
MAX_LEN    = 256    # 每条评论截取的最大 token 数
BATCH_SIZE = 16
LR         = 2e-5   # BERT 微调的经典学习率
EPOCHS     = 3      # BERT 通常 3 个 epoch 就收敛

log(f'\n设备：{DEVICE}')
log(f'模型：{MODEL_NAME}')
log(f'最大序列长度：{MAX_LEN}  Batch：{BATCH_SIZE}  LR：{LR}\n')

# -------- 1. 加载数据集 --------
log('加载 IMDb 数据集...')
dataset = load_dataset('imdb')
# dataset['train'] 有 25,000 条，dataset['test'] 有 25,000 条
# 每条有两个字段：text（评论文本）和 label（0=负面，1=正面）

# 为了加快演示速度，先各取 2000 条
# 正式训练时把这两行删掉，用全量数据
train_data = dataset['train'].shuffle(seed=42).select(range(2000))
test_data  = dataset['test'].shuffle(seed=42).select(range(500))

log(f'训练集：{len(train_data)} 条，测试集：{len(test_data)} 条\n')

# -------- 2. 加载 Tokenizer --------
# Tokenizer 的作用：把原始文字变成模型能理解的数字序列
# 例如："I love this movie" →
#   input_ids:      [101, 1045, 2293, 2023, 3185, 102]
#   attention_mask: [1,   1,    1,    1,    1,    1  ]
log('加载 Tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',    # 不足 MAX_LEN 的用 [PAD] 补齐
        truncation=True,         # 超过 MAX_LEN 的截断
        max_length=MAX_LEN,
        return_tensors='pt'
    )

# 对整个数据集批量做 tokenize
log('Tokenizing 数据集（第一次运行需要约一分钟）...')
train_enc = train_data.map(tokenize, batched=True, batch_size=64)
test_enc  = test_data.map(tokenize, batched=True, batch_size=64)

# 设置 PyTorch 格式，只保留模型需要的列
train_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_enc.set_format('torch',  columns=['input_ids', 'attention_mask', 'label'])

trainloader = DataLoader(train_enc, batch_size=BATCH_SIZE, shuffle=True)
testloader  = DataLoader(test_enc,  batch_size=BATCH_SIZE, shuffle=False)

# -------- 3. 加载预训练模型 --------
log(f'\n加载预训练模型 {MODEL_NAME}...')
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2    # 两类：正面 / 负面
)
model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
log(f'模型总参数量：{total_params:,}')
log(f'（大部分是从 HuggingFace 下载的预训练权重）\n')

# -------- 4. 优化器 + 学习率调度 --------
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps  = len(trainloader) * EPOCHS
warmup_steps = total_steps // 10   # 前 10% 的步骤做 warmup

# ================================================================
# warmup：学习率从 0 线性增加到 LR，再按余弦曲线下降
# 这是 BERT 系列模型微调的标配，防止一开始学习率太大破坏预训练权重
# ================================================================
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# -------- 5. 评估函数 --------
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in testloader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    model.train()
    return 100 * correct / total

# -------- 6. 训练循环 --------
log('开始训练...')
log('-' * 60)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(trainloader):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)

        optimizer.zero_grad()

        # HuggingFace 的模型直接接收 labels，内部计算 CrossEntropyLoss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()

        # 梯度裁剪：防止梯度爆炸（BERT 微调的标配）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(trainloader)
    acc = evaluate()

    # ================================================================
    # 【TXT日志】每轮结果
    # ================================================================
    msg = (f'Epoch [{epoch+1}/{EPOCHS}]  '
           f'Loss: {avg_loss:.4f}  '
           f'准确率: {acc:.2f}%')
    log(msg)
    if acc > best_acc:
        best_acc = acc
        model.save_pretrained('./best_sentiment_model')
        tokenizer.save_pretrained('./best_sentiment_model')
        log(f'  ★ 最佳模型已保存！准确率：{best_acc:.2f}%')
    # ================================================================

log('-' * 60)
log(f'\n训练完成！最佳准确率：{best_acc:.2f}%')

# -------- 7. 用最佳模型预测几条真实评论 --------
log('\n--- 实际预测示例 ---')
examples = [
    "This movie is absolutely fantastic! The acting is superb.",
    "Terrible film. Boring plot, bad acting. Complete waste of time.",
    "It was okay, not great but not terrible either.",
]

from transformers import pipeline
classifier = pipeline(
    'sentiment-analysis',
    model='./best_sentiment_model',
    device=0 if torch.cuda.is_available() else -1
)

for text in examples:
    result = classifier(text)[0]
    label = 'POSITIVE' if result['label'] == 'LABEL_1' else 'NEGATIVE'
    log(f'  [{label}] ({result["score"]:.2%}) {text[:60]}...')

# ================================================================
log('=' * 60)
log(f'结束时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'日志：{log_path}')
log('=' * 60)
# ================================================================
