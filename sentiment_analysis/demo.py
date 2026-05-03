import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- 加载模型 --------
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('./best_sentiment_model')
model     = AutoModelForSequenceClassification.from_pretrained(
    './best_sentiment_model').to(DEVICE)
model.eval()

def predict(text):
    """输入一段文字，返回情感标签和置信度"""
    if not text.strip():
        return "请输入文字", {}

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]

    neg_prob = probs[0].item()
    pos_prob = probs[1].item()

    label = "🟢 POSITIVE（正面）" if pos_prob > neg_prob else "🔴 NEGATIVE（负面）"

    # 返回标签字符串和置信度字典（Gradio 会自动画柱状图）
    return label, {"POSITIVE": round(pos_prob, 4), "NEGATIVE": round(neg_prob, 4)}


# -------- 构建 Gradio 界面 --------
with gr.Blocks(title="情感分析 Demo") as demo:
    gr.Markdown("## 🎬 电影评论情感分析")
    gr.Markdown("输入一段英文影评，模型会判断它是正面还是负面。")

    with gr.Row():
        input_box = gr.Textbox(
            label="输入评论",
            placeholder="例如：This movie was absolutely wonderful!",
            lines=4
        )

    btn = gr.Button("分析", variant="primary")

    with gr.Row():
        label_out = gr.Label(label="预测结果")
        prob_out  = gr.Label(label="置信度")

    # 预设几个例子，点一下就能填入
    gr.Examples(
        examples=[
            ["This movie is absolutely fantastic! Best film I've seen this year."],
            ["Terrible waste of time. The plot makes no sense and acting is awful."],
            ["It was okay, not great but not terrible either."],
            ["I can't believe how bad this was. I want my two hours back."],
            ["A masterpiece of modern cinema. Every scene is breathtaking."],
        ],
        inputs=input_box
    )

    btn.click(fn=predict, inputs=input_box, outputs=[label_out, prob_out])

# 启动：在终端里会显示一个本地链接，用浏览器打开即可
demo.launch()
