import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

os.makedirs('./transformer_logs', exist_ok=True)

# ================================================================
# 把之前写好的所有组件复制进来（原封不动）
# ================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        Q = self.W_Q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_O(out), weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        x = self.pos_encoding(self.embedding(x))
        all_weights = []
        for layer in self.layers:
            x, w = layer(x, mask)
            all_weights.append(w)
        return self.norm(x), all_weights


# ================================================================
# 第一部分：从零构建词表
# ================================================================

print("=" * 60)
print("  第六阶段：从零构建词表 + 训练文本分类器")
print("=" * 60)

# 数据集：精心挑选的正/负面句子
# 特意让某些词明确携带情感信号，让注意力有东西可以学
TRAIN_DATA = [
    # ---- POSITIVE (label=1) ----
    ("this film is absolutely wonderful and touching",       1),
    ("a brilliant masterpiece with stunning performances",   1),
    ("fantastic story i loved every moment of it",          1),
    ("the acting is superb and the plot is engaging",       1),
    ("one of the best movies i have ever seen",             1),
    ("beautiful visuals and a deeply moving story",         1),
    ("i love this movie it is truly amazing",               1),
    ("outstanding direction and wonderful performances",    1),
    ("a perfect film with great characters",                1),
    ("incredible and emotional highly recommend this",      1),
    ("this is a great film with beautiful scenery",         1),
    ("loved the story and the brilliant cast",              1),
    ("wonderful film deeply touching and inspiring",        1),
    ("amazing performances make this a must watch",         1),
    ("the best film this year absolutely loved it",         1),
    ("superb acting and a fantastic engaging story",        1),
    ("this movie made me feel wonderful and happy",         1),
    ("a truly great and moving cinematic experience",       1),
    ("brilliant direction and a touching love story",       1),
    ("loved every scene this is an outstanding film",       1),
    ("what a wonderful and uplifting movie experience",     1),
    ("truly brilliant film with amazing performances",      1),
    ("i absolutely loved this heartwarming great film",     1),
    ("a fantastic and beautiful movie loved every bit",     1),
    ("superb and wonderful highly recommend this film",     1),
    ("this touching story is truly great and inspiring",    1),
    ("outstanding and brilliant a wonderful movie",         1),
    ("loved it completely a perfect heartwarming film",     1),
    ("beautiful and amazing this film is truly superb",     1),
    ("a great film with brilliant and touching moments",    1),
    ("fantastic performances in this wonderful movie",      1),
    ("deeply moving and beautiful i loved this film",       1),
    ("amazing story brilliant acting great direction",      1),
    ("this is truly wonderful and a must see film",         1),
    ("outstanding movie loved every wonderful moment",      1),
    ("brilliant film beautiful story amazing cast loved",   1),
    ("a superb and touching film truly loved it",           1),
    ("wonderful direction and a fantastic moving story",    1),
    ("great film absolutely brilliant and deeply moving",   1),
    ("loved this amazing film outstanding in every way",    1),
    # ---- NEGATIVE (label=0) ----
    ("this movie is terrible and a complete waste",         0),
    ("awful film boring plot and bad acting throughout",    0),
    ("i hated this movie it was so disappointing",         0),
    ("horrible direction and the worst cast ever seen",    0),
    ("a dreadful film i walked out halfway through",       0),
    ("terrible script and completely pointless story",     0),
    ("boring and stupid i want my money back now",         0),
    ("the worst movie i have ever had to watch",           0),
    ("awful and dull completely wasted my time",           0),
    ("this film is bad and the acting is atrocious",       0),
    ("a terrible disappointment boring and poorly made",   0),
    ("hated every minute this is a dreadful film",         0),
    ("completely awful and the worst film this year",      0),
    ("bad acting and a horrible confusing plot",           0),
    ("terrible movie do not waste your time watching",     0),
    ("a boring and dreadful film with no redeeming",       0),
    ("this was awful i hated the characters",              0),
    ("worst film ever made terrible acting and story",     0),
    ("so bad and boring i almost fell asleep",             0),
    ("dreadful and pointless avoid this terrible film",    0),
    ("what a horrible and dreadful waste of time",         0),
    ("truly awful film with terrible performances",        0),
    ("i completely hated this boring terrible movie",      0),
    ("a dreadful and horrible movie hated every bit",      0),
    ("awful and terrible highly avoid this film",          0),
    ("this horrible story is truly bad and boring",        0),
    ("dreadful and terrible a horrible movie",             0),
    ("hated it completely a pointless dreadful film",      0),
    ("horrible and awful this film is truly terrible",     0),
    ("a bad film with awful and terrible moments",         0),
    ("terrible performances in this horrible movie",       0),
    ("deeply boring and awful i hated this film",          0),
    ("awful story terrible acting bad direction",          0),
    ("this is truly horrible and a waste of time",         0),
    ("dreadful movie hated every horrible moment",         0),
    ("terrible film horrible story awful cast hated",      0),
    ("a awful and horrible film truly hated it",           0),
    ("terrible direction and a dreadful boring story",     0),
    ("bad film absolutely horrible and deeply boring",     0),
    ("hated this awful film dreadful in every way",        0),
]

TEST_DATA = [
    ("a wonderful and touching film i loved it",           1),
    ("brilliant and amazing truly a great movie",          1),
    ("outstanding performances in a beautiful story",      1),
    ("loved this fantastic and deeply moving film",        1),
    ("a terrible and boring film i hated it",              0),
    ("awful movie horrible acting complete waste",         0),
    ("the worst and most dreadful film ever made",         0),
    ("hated this terrible and boring awful movie",         0),
]


# ----------------------------------------------------------------
# 从零构建词表
# 词表 = {词: ID}，PAD=0，UNK=1，然后按频率排列
# ----------------------------------------------------------------
print("\n【步骤 1】从零构建词表\n")

all_words = []
for text, _ in TRAIN_DATA:
    all_words.extend(text.lower().split())

word_counts = Counter(all_words)
# 按频率从高到低排序
sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

vocab = {'<PAD>': 0, '<UNK>': 1}
for word, count in sorted_words:
    vocab[word] = len(vocab)

VOCAB_SIZE = len(vocab)
print(f"词表大小：{VOCAB_SIZE} 个词")
print(f"词表示例（前10个）：{list(vocab.items())[:10]}")

# 显示情感词的 ID
key_words = ['wonderful', 'brilliant', 'terrible', 'awful', 'loved', 'hated']
print(f"\n关键情感词的 ID：")
for w in key_words:
    print(f"  '{w}' → ID {vocab.get(w, 1)}")


# ----------------------------------------------------------------
# 把句子转成 ID 序列，并做 padding（补齐到同样长度）
# ----------------------------------------------------------------
def encode(text, vocab, max_len=20):
    """把一句话变成固定长度的 ID 列表"""
    ids = [vocab.get(w, vocab['<UNK>']) for w in text.lower().split()]
    # 超长截断，不足补 PAD
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return ids

MAX_LEN = 20

# 验证一个例子
sample_text = "this film is absolutely wonderful"
sample_ids = encode(sample_text, vocab, MAX_LEN)
print(f"\n句子编码示例：")
print(f"  文本：'{sample_text}'")
print(f"  ID序列：{sample_ids}")
print(f"  （0 表示 PAD 填充）")


# ================================================================
# 第二部分：Dataset + DataLoader
# ================================================================

class SentimentDataset(Dataset):
    def __init__(self, data, vocab, max_len=20):
        self.samples = []
        for text, label in data:
            ids = encode(text, vocab, max_len)
            self.samples.append((torch.tensor(ids), torch.tensor(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

train_dataset = SentimentDataset(TRAIN_DATA, vocab, MAX_LEN)
test_dataset  = SentimentDataset(TEST_DATA,  vocab, MAX_LEN)
train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=6, shuffle=False)

print(f"\n【步骤 2】数据集准备完成")
print(f"  训练集：{len(train_dataset)} 条")
print(f"  测试集：{len(test_dataset)} 条")


# ================================================================
# 第三部分：模型 = Transformer Encoder + 分类头
# ================================================================

class TransformerClassifier(nn.Module):
    """
    完整的分类模型：
    Encoder 负责理解句子 → 分类头负责做判断

    关键设计：用 [CLS] token 的输出做分类
    BERT 就是这样做的：在句子最前面加一个特殊标记 [CLS]，
    训练完之后，[CLS] 位置的向量代表了整个句子的语义。
    我们这里用"第一个 token 的平均"来简化。
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers,
            d_ff, max_len, dropout
        )
        # 分类头：把 d_model 维向量映射到 num_classes 个类别的得分
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len]
        encoded, attn_weights = self.encoder(x)
        # encoded: [batch, seq_len, d_model]

        # 池化策略：对所有非 PAD 位置的向量取平均
        # 这比只取第一个位置更稳定
        # x==0 的地方是 PAD，不参与平均
        mask = (x != 0).unsqueeze(-1).float()      # [batch, seq_len, 1]
        pooled = (encoded * mask).sum(dim=1) \
                 / mask.sum(dim=1).clamp(min=1)   # [batch, d_model]

        logits = self.classifier(pooled)           # [batch, num_classes]
        return logits, attn_weights


# ================================================================
# 第四部分：保存训练前的注意力权重，用于对比
# ================================================================

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL    = 64       # 从 128 缩小到 64
NUM_HEADS  = 4
NUM_LAYERS = 2        # 从 3 层减到 2 层
D_FF       = 256
DROPOUT    = 0.1
EPOCHS     = 80       # 增加到 80 epochs
LR         = 5e-4
WARMUP     = 10       # 前 10 个 epoch 线性升温


model = TransformerClassifier(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    num_classes=2,
    max_len=MAX_LEN,
    dropout=DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n【步骤 3】模型初始化完成，参数量：{total_params:,}\n")

# 挑一个有明确情感信号的测试句子
PROBE_TEXT = "this film is absolutely wonderful"
probe_ids  = torch.tensor([encode(PROBE_TEXT, vocab, MAX_LEN)]).to(DEVICE)
probe_words = PROBE_TEXT.split()

def get_attention_layer1(m, ids):
    """取第一层、第一个头的注意力矩阵（seq_len × seq_len）"""
    m.eval()
    with torch.no_grad():
        _, attn_list = m(ids)
    # attn_list[0]: [batch=1, heads=4, seq_len, seq_len]
    # 取 head=0，去掉 batch 维
    return attn_list[0][0, 0, :len(probe_words), :len(probe_words)].cpu().numpy()

weights_before = get_attention_layer1(model, probe_ids)
print(f"探针句子：'{PROBE_TEXT}'")
print(f"训练前注意力矩阵（第1层第1头，{len(probe_words)}×{len(probe_words)}）：")
print(np.round(weights_before, 3))


# ================================================================
# 第五部分：训练循环
# ================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

# Warmup + 余弦退火：前 WARMUP 个 epoch 线性升温，之后余弦下降
def lr_lambda(epoch):
    if epoch < WARMUP:
        return (epoch + 1) / WARMUP          # 线性升温
    # 余弦退火
    progress = (epoch - WARMUP) / (EPOCHS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

loss_fn    = nn.CrossEntropyLoss()

print(f"\n{'='*60}")
print(f"  开始训练  ({EPOCHS} epochs, lr={LR}, device={DEVICE})")
print(f"{'='*60}")

train_losses, train_accs = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for ids, labels in train_loader:
        ids, labels = ids.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(ids)
        loss = loss_fn(logits, labels)
        loss.backward()
        # 梯度裁剪：防止梯度爆炸（Transformer 训练的常用技巧）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    acc      = 100 * correct / total
    train_losses.append(avg_loss)
    train_accs.append(acc)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:>2}/{EPOCHS}]  "
              f"Loss: {avg_loss:.4f}  Acc: {acc:.1f}%")


# ================================================================
# 第六部分：测试集评估
# ================================================================

model.eval()
correct, total = 0, 0
print(f"\n{'='*60}")
print("  测试集预测结果")
print(f"{'='*60}")

with torch.no_grad():
    for ids, labels in test_loader:
        ids, labels = ids.to(DEVICE), labels.to(DEVICE)
        logits, _ = model(ids)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # 打印每条测试句子的预测
        for i in range(ids.size(0)):
            text, true_label = TEST_DATA[i]
            pred  = preds[i].item()
            prob  = torch.softmax(logits[i], dim=0)
            label_map = {0: 'NEGATIVE', 1: 'POSITIVE'}
            mark = '✓' if pred == labels[i].item() else '✗'
            print(f"  [{mark}] 真实:{label_map[true_label]}  "
                  f"预测:{label_map[pred]}  "
                  f"置信度:{prob[pred].item():.1%}  "
                  f"'{text[:40]}'")

print(f"\n测试准确率：{100*correct/total:.1f}%")


# ================================================================
# 第七部分：训练后注意力 + 对比可视化
# ================================================================

weights_after = get_attention_layer1(model, probe_ids)
print(f"\n训练后注意力矩阵（第1层第1头）：")
print(np.round(weights_after, 3))

# 绘制对比图
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

def draw_attn(ax, matrix, words, title):
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(words, fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    for i in range(len(words)):
        for j in range(len(words)):
            val = matrix[i, j]
            c = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.5, color=c)
    return im

draw_attn(axes[0], weights_before, probe_words,
          f'Before Training\n"{PROBE_TEXT}"')
draw_attn(axes[1], weights_after,  probe_words,
          f'After Training ({EPOCHS} epochs)\n"{PROBE_TEXT}"')

# 第三张图：差值图（训练后 - 训练前）
diff = weights_after - weights_before
im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[2].set_xticks(range(len(probe_words)))
axes[2].set_yticks(range(len(probe_words)))
axes[2].set_xticklabels(probe_words, rotation=45, ha='right', fontsize=9)
axes[2].set_yticklabels(probe_words, fontsize=9)
axes[2].set_title('Change (After - Before)\nRed=increased  Blue=decreased',
                   fontsize=11, pad=8)
for i in range(len(probe_words)):
    for j in range(len(probe_words)):
        val = diff[i, j]
        axes[2].text(j, i, f'{val:+.2f}', ha='center', va='center',
                     fontsize=7.5, color='black')

plt.colorbar(im3, ax=axes[2])
plt.suptitle(f'Attention Weight Change After Training\nProbe: "{PROBE_TEXT}"',
             fontsize=13, y=1.02)
plt.tight_layout()
compare_path = './transformer_logs/attention_before_after.png'
plt.savefig(compare_path, dpi=150, bbox_inches='tight')
print(f"\n对比图已保存：{compare_path}")


# 绘制训练曲线
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(1, EPOCHS+1), train_losses, 'b-o', markersize=3)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(alpha=0.3)

ax2.plot(range(1, EPOCHS+1), train_accs, 'g-o', markersize=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy')
ax2.set_ylim(0, 105)
ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100%')
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
curve_path = './transformer_logs/training_curve.png'
plt.savefig(curve_path, dpi=150)
print(f"训练曲线已保存：{curve_path}")

print(f"\n{'='*60}")
print("  训练完成！你已经从零实现并训练了一个完整的 Transformer")
print(f"{'='*60}")
