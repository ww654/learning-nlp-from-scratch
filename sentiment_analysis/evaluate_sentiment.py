import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import os
from datetime import datetime
# 用于绘图
import matplotlib
matplotlib.use('Agg')   # 无图形界面时也能保存图片
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)

# ================================================================
# 【TXT日志】初始化
# ================================================================
os.makedirs('./logs', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f'./logs/eval_{timestamp}.txt'

def log(msg):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

log('=' * 60)
log('  IMDb 情感分析 — 深度评估')
log(f'  开始时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log('=' * 60)
# ================================================================

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN  = 256
BATCH    = 32

# -------- 1. 加载测试集（取 1000 条足够分析） --------
dataset  = load_dataset('imdb')
test_raw = dataset['test'].shuffle(seed=42).select(range(1000))

tokenizer = AutoTokenizer.from_pretrained('./best_sentiment_model')
model     = AutoModelForSequenceClassification.from_pretrained(
    './best_sentiment_model').to(DEVICE)
model.eval()

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length', truncation=True,
        max_length=MAX_LEN, return_tensors='pt'
    )

test_enc = test_raw.map(tokenize, batched=True, batch_size=64)
test_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
testloader = DataLoader(test_enc, batch_size=BATCH, shuffle=False)

# -------- 2. 收集预测结果 --------
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for batch in testloader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1)
        preds   = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

# ================================================================
# 【TXT日志】分类报告
# ================================================================
report = classification_report(
    all_labels, all_preds,
    target_names=['NEGATIVE', 'POSITIVE'], digits=4
)
log('\n分类报告：')
log(report)
overall_acc = 100 * np.sum(all_preds == all_labels) / len(all_labels)
log(f'总体准确率：{overall_acc:.2f}%\n')
# ================================================================

# -------- 3. 混淆矩阵 --------
cm = confusion_matrix(all_labels, all_preds)

# ================================================================
# 【TXT日志】混淆矩阵数字
# ================================================================
log('混淆矩阵：')
log('              预测NEGATIVE  预测POSITIVE')
log(f'真实NEGATIVE  {cm[0,0]:>10}   {cm[0,1]:>10}')
log(f'真实POSITIVE  {cm[1,0]:>10}   {cm[1,1]:>10}')
# ================================================================

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=['NEGATIVE', 'POSITIVE']).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — IMDb Sentiment')
plt.tight_layout()
cm_path = f'./logs/confusion_matrix_{timestamp}.png'
plt.savefig(cm_path, dpi=150)
log(f'\n混淆矩阵已保存：{cm_path}')

# -------- 4. ROC 曲线 --------
fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, label=f'DistilBERT (AUC = {roc_auc:.4f})')
ax.plot([0,1],[0,1], 'k--', label='随机猜测 (AUC = 0.50)')
ax.set_xlabel('假正率 (FPR)')
ax.set_ylabel('真正率 (TPR)')
ax.set_title('ROC Curve — IMDb Sentiment')
ax.legend()
plt.tight_layout()
roc_path = f'./logs/roc_curve_{timestamp}.png'
plt.savefig(roc_path, dpi=150)

# ================================================================
# 【TXT日志】AUC
# ================================================================
log(f'\nAUC = {roc_auc:.4f}')
log(f'ROC 曲线已保存：{roc_path}')
# ================================================================

# -------- 5. 最有价值的部分：找出判错的例子 --------
# 找出"非常自信但判错了"的样本（置信度 > 85% 却预测错误）
wrong_indices = np.where(all_preds != all_labels)[0]
texts = test_raw['text']
labels_raw = test_raw['label']

log('\n' + '=' * 60)
log('模型判错的例子（置信度 > 85%）：')
log('=' * 60)

shown = 0
for idx in wrong_indices:
    confidence = max(all_probs[idx])
    if confidence > 0.85 and shown < 5:
        true_label = 'POSITIVE' if labels_raw[idx] == 1 else 'NEGATIVE'
        pred_label = 'POSITIVE' if all_preds[idx] == 1 else 'NEGATIVE'
        # 只显示前 200 个字符
        preview = texts[idx][:200].replace('\n', ' ')
        log(f'\n--- 样本 {idx} ---')
        log(f'真实标签：{true_label}')
        log(f'预测标签：{pred_label}  置信度：{confidence:.1%}')
        log(f'文本预览：{preview}...')
        shown += 1

# ================================================================
# 【TXT日志】收尾
# ================================================================
log('\n' + '=' * 60)
log(f'评估结束：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'完整日志：{log_path}')
log('=' * 60)
# ================================================================
