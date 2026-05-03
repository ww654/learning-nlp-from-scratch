# Learning NLP From Scratch

A hands-on learning journey through modern NLP — from fine-tuning pre-trained models to implementing a Transformer from the ground up in pure PyTorch.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## Projects

### [01 · Sentiment Analysis with DistilBERT](./01_sentiment_analysis/)

Fine-tuned `distilbert-base-uncased` on the IMDb dataset for binary sentiment classification. Trained on 25,000 reviews, evaluated with confusion matrix, ROC curve, and error analysis.

**Result: 91.15% accuracy · AUC 0.9509**

| Metric | Small Data (2k) | Full Data (25k) |
|--------|:-:|:-:|
| Accuracy | 87.0% | **91.15%** |
| AUC | — | **0.9509** |
| Loss | 0.17 | 0.17 |

Key finding: the model handles direct sentiment well but struggles with **sarcasm** — sentences like *"A truly masterful piece of filmmaking. It managed to put me to sleep."* are confidently misclassified as positive (96.6% confidence).

---

### [02 · Transformer Encoder From Scratch](./02_transformer_from_scratch/)

A complete Transformer Encoder implemented in pure PyTorch — no HuggingFace, no shortcuts. Every component written and explained from first principles.

**Components implemented:**
- Scaled dot-product attention
- Multi-head attention (4 heads, parallel)
- Sinusoidal positional encoding
- Feed-forward network (with ReLU)
- Residual connections + Layer Normalization
- Full Encoder stack (2 layers)

**Training result:** Loss dropped from 0.72 → 0.01, accuracy converged to 100% on training set.

The most interesting result — **attention weights before vs. after training:**

![Attention Before/After](./02_transformer_from_scratch/results/attention_before_after.png)

After training, `"wonderful" → "film"` became the strongest single connection (+0.15), showing the model learned that sentiment adjectives anchor to the noun they describe.

---

## What I Learned

**Why Transformers replaced RNNs**
RNNs process words sequentially and lose long-range context. Transformers let every word attend to every other word simultaneously — no information bottleneck, fully parallelizable.

**Why pre-training matters**
When training from scratch on only 40 sentences, the model's loss stuck at 0.693 (= ln 2, i.e. random guessing). The same architecture fine-tuned from BERT's pre-trained weights reached 91% accuracy. Data scale is everything for Transformers.

**What learning rate warmup does**
Without warmup: loss oscillated around 0.693 for 30 epochs, never converging.
With warmup: loss broke through at epoch ~18 and collapsed to near zero. The "plateau then cliff" shape in the training curve is the warmup signature.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/learning-nlp-from-scratch.git
cd learning-nlp-from-scratch

pip install torch transformers datasets scikit-learn matplotlib gradio
```

The IMDb dataset is downloaded automatically via HuggingFace `datasets`. If you are in a region with restricted access, set the mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Stack

| Library | Purpose |
|---------|---------|
| `torch` | Model implementation and training |
| `transformers` | DistilBERT tokenizer and model |
| `datasets` | IMDb dataset loading |
| `scikit-learn` | Metrics: confusion matrix, ROC, classification report |
| `matplotlib` | All visualizations |
| `gradio` | Interactive demo UI |
