import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

os.makedirs('./transformer_logs', exist_ok=True)

print("=" * 60)
print("  从零实现 Transformer")
print("=" * 60)

# ================================================================
# 第一部分：缩放点积注意力
# Scaled Dot-Product Attention
# ================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    参数：
        Q: Query 矩阵，形状 [batch, heads, seq_len, d_k]
        K: Key   矩阵，形状 [batch, heads, seq_len, d_k]
        V: Value 矩阵，形状 [batch, heads, seq_len, d_k]
        mask: 可选的掩码矩阵（用于 Decoder，Encoder 不需要）
    返回：
        output:  注意力加权后的输出，形状 [batch, heads, seq_len, d_k]
        weights: 注意力权重矩阵，形状 [batch, heads, seq_len, seq_len]
                 （可视化时用这个）
    """
    d_k = Q.size(-1)  # 取最后一维，即向量维度

    # 第一步：计算相关度分数
    # Q: [..., seq_len, d_k] × K^T: [..., d_k, seq_len] → [..., seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1))  # K.transpose(-2,-1) 即 K^T

    # 第二步：缩放，防止梯度消失
    scores = scores / math.sqrt(d_k)

    # 第三步（可选）：应用掩码
    # mask 里值为 True 的位置会被设成负无穷，Softmax 后变成 0
    # 这样模型就"看不到"那些被遮住的位置
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 第四步：Softmax，把分数变成概率
    weights = F.softmax(scores, dim=-1)

    # 第五步：用权重对 V 做加权求和
    output = torch.matmul(weights, V)

    return output, weights


# ================================================================
# 验证：用一个小例子手动感受一下注意力在做什么
# ================================================================

print("\n【验证】用简单数字感受注意力机制\n")

# 假设我们有一个句子，4 个词，每个词用 4 维向量表示
# 为了直观，我们手动造几个"有意义"的向量
torch.manual_seed(42)

# batch=1, heads=1（先不考虑多头）, seq_len=4, d_k=4
batch, heads, seq_len, d_k = 1, 1, 4, 4

# 造假数据：假设这 4 个词是 ["I", "love", "this", "movie"]
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"输入 Q 的形状：{Q.shape}")
print(f"输入 K 的形状：{K.shape}")
print(f"输入 V 的形状：{V.shape}")
print(f"\n注意力权重矩阵形状：{weights.shape}")
print(f"注意力权重（去掉 batch 和 heads 维度）：")
w = weights.squeeze(0).squeeze(0)  # 变成 [4, 4]
print(w.detach().numpy().round(3))
print("\n每行之和（应该都等于 1.0，这是 Softmax 的性质）：")
print(w.sum(dim=-1).detach().numpy().round(6))
print(f"\n输出形状：{output.shape}  （和输入形状相同）")


# ================================================================
# 可视化：把注意力权重矩阵画出来（纯英文标签，兼容所有系统）
# ================================================================

words = ["I", "love", "this", "movie"]

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(w.detach().numpy(), cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(words, fontsize=12)
ax.set_yticklabels(words, fontsize=12)
ax.set_xlabel("Key  (word being attended to)", fontsize=11)
ax.set_ylabel("Query  (word currently being processed)", fontsize=11)
ax.set_title("Attention Weight Matrix\n(darker = higher attention)", fontsize=12)
plt.colorbar(im, ax=ax)

for i in range(4):
    for j in range(4):
        val = w[i, j].item()
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=10, color=color)

plt.tight_layout()
path = './transformer_logs/attention_weights.png'
plt.savefig(path, dpi=150)
print(f"Saved: {path}")

# ================================================================
# 第二部分：多头注意力 + 位置编码 + FFN + 残差 + LayerNorm
# 把所有组件组装成一个完整的 Transformer Encoder Block
# ================================================================

print("\n" + "=" * 60)
print("  第三阶段：多头注意力 + 完整 Encoder Block")
print("=" * 60)


# ----------------------------------------------------------------
# 2.1 多头注意力层
# ----------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model:   词向量总维度（比如 512）
        num_heads: 注意力头的数量（比如 8）
        每个头的维度 d_k = d_model // num_heads = 64
        """
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads  # 每个头的维度

        # 四个线性层：Q、K、V 的投影 + 最后的输出投影
        # 注意：这里用一个大矩阵同时处理所有头，效率更高
        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # [d_model → d_model]
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)  # 输出投影

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # ------ 第一步：线性投影 ------
        # x: [batch, seq_len, d_model]
        # 经过线性层后还是 [batch, seq_len, d_model]
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # ------ 第二步：切分成多个头 ------
        # 把 d_model 维度拆成 [num_heads, d_k]
        # reshape: [batch, seq_len, d_model] → [batch, seq_len, num_heads, d_k]
        # transpose: → [batch, num_heads, seq_len, d_k]
        # 这样每个头可以独立并行计算注意力
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 现在 Q/K/V 形状都是 [batch, num_heads, seq_len, d_k]

        # ------ 第三步：对每个头计算注意力 ------
        # scaled_dot_product_attention 是我们之前写的函数
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: [batch, num_heads, seq_len, d_k]

        # ------ 第四步：把各头结果拼回来 ------
        # transpose: [batch, num_heads, seq_len, d_k] → [batch, seq_len, num_heads, d_k]
        # contiguous + view: → [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # ------ 第五步：输出线性投影 ------
        output = self.W_O(attn_output)  # [batch, seq_len, d_model]

        return output, attn_weights


# ----------------------------------------------------------------
# 2.2 位置编码
# ----------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        d_model: 词向量维度
        max_len: 支持的最大句子长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预先计算好所有位置的编码，存成固定矩阵
        # pe 形状：[max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]

        # 分母：10000^(2i/d_model)，用 exp(log) 形式避免数值不稳定
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 0,2,4,...
        pe[:, 1::2] = torch.cos(position * div_term)  # 1,3,5,...

        # 加一个 batch 维度，变成 [1, max_len, d_model]
        # 这样可以直接广播加到 [batch, seq_len, d_model] 上
        pe = pe.unsqueeze(0)

        # register_buffer：不是模型参数（不会被优化器更新），
        # 但会随模型一起保存和移动（.to(device) 时自动跟着走）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # self.pe[:, :x.size(1)] 取前 seq_len 个位置的编码
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ----------------------------------------------------------------
# 2.3 前馈网络（FFN）
# ----------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: 输入输出维度（比如 512）
        d_ff:    中间层维度（通常是 4 * d_model = 2048）
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 512 → 2048
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),   # 2048 → 512
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------
# 2.4 Encoder Block（一层）
# 结构：输入 → [多头注意力 + 残差 + LayerNorm] → [FFN + 残差 + LayerNorm] → 输出
# ----------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention  = MultiHeadAttention(d_model, num_heads)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)  # 注意力之后的 LayerNorm
        self.norm2      = nn.LayerNorm(d_model)  # FFN 之后的 LayerNorm
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ------ 子层 1：多头注意力 + 残差 + LayerNorm ------
        attn_out, attn_weights = self.attention(x, mask)
        # 残差连接：x + Sublayer(x)，然后 LayerNorm
        x = self.norm1(x + self.dropout(attn_out))

        # ------ 子层 2：FFN + 残差 + LayerNorm ------
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


# ----------------------------------------------------------------
# 2.5 完整的 Transformer Encoder
# 把 N 个 EncoderBlock 堆叠起来
# ----------------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_len=512, dropout=0.1):
        """
        vocab_size:  词表大小（有多少个不同的词）
        d_model:     词向量维度
        num_heads:   注意力头数
        num_layers:  堆叠几个 EncoderBlock（原论文是 6）
        d_ff:        FFN 中间层维度
        max_len:     支持的最大句子长度
        """
        super().__init__()
        # 词嵌入：把词的 ID（整数）变成 d_model 维向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 堆叠 N 个 Encoder Block
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)  # 最后一层归一化

        # 参数初始化（用 Xavier 均匀初始化，比随机初始化收敛更稳）
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len]，每个元素是词的 ID（整数）
        """
        # 词嵌入 + 位置编码
        x = self.embedding(x)           # [batch, seq_len, d_model]
        x = self.pos_encoding(x)        # 加入位置信息

        # 逐层通过 Encoder Block
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        x = self.norm(x)
        return x, all_attn_weights


# ================================================================
# 验证：实例化完整 Encoder，检查形状是否符合预期
# ================================================================

print("\n【验证】实例化完整 Transformer Encoder\n")

# 用小参数方便调试
VOCAB_SIZE  = 1000   # 词表 1000 个词
D_MODEL     = 128    # 向量维度（原论文 512，这里缩小方便运行）
NUM_HEADS   = 8      # 注意力头数（d_model 必须被整除：128/8=16）
NUM_LAYERS  = 4      # 堆叠 4 层（原论文 6）
D_FF        = 512    # FFN 中间层（原论文 2048，缩小比例保持 4x）
MAX_LEN     = 64     # 最大句子长度

model = TransformerEncoder(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN,
)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量：{total_params:,}")
print(f"（DistilBERT 是 66,955,010 个参数，我们的 mini 版是它的"
      f" {total_params/66955010*100:.1f}%）\n")

# 造一个假输入：batch=2，每个句子 10 个词，词 ID 随机
batch_size = 2
seq_len    = 10
fake_input = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

print(f"输入形状：{fake_input.shape}  （batch=2, 每句 10 个词）")

with torch.no_grad():
    output, all_weights = model(fake_input)

print(f"输出形状：{output.shape}  （batch=2, 每句 10 个词, 每词 128 维向量）")
print(f"注意力权重层数：{len(all_weights)}  （对应 {NUM_LAYERS} 个 Block）")
print(f"每层注意力权重形状：{all_weights[0].shape}")
print(f"  → [batch={batch_size}, heads={NUM_HEADS},"
      f" seq={seq_len}, seq={seq_len}]")


# ================================================================
# 可视化：展示 4 层 × 8 头的注意力模式
# 用热力图看不同的头学到了什么不同的关注模式
# ================================================================

print("\n生成多头注意力可视化图...")

fig, axes = plt.subplots(NUM_LAYERS, NUM_HEADS,
                          figsize=(NUM_HEADS * 2.2, NUM_LAYERS * 2.2))
fig.suptitle(
    f"All Attention Heads  ({NUM_LAYERS} layers × {NUM_HEADS} heads)\n"
    "Each cell = attention weight matrix for one head",
    fontsize=12
)

for layer_idx in range(NUM_LAYERS):
    for head_idx in range(NUM_HEADS):
        ax = axes[layer_idx][head_idx]
        # 取第 0 个样本，第 layer_idx 层，第 head_idx 头
        w_matrix = all_weights[layer_idx][0, head_idx].detach().numpy()
        ax.imshow(w_matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"L{layer_idx+1}-H{head_idx+1}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
multi_path = './transformer_logs/multi_head_attention.png'
plt.savefig(multi_path, dpi=120)
print(f"多头注意力可视化已保存：{multi_path}")

print("\n" + "=" * 60)
print("  所有组件验证通过！形状符合预期。")
print("  下一步：第六阶段，在真实任务上训练这个 Encoder")
print("=" * 60)
