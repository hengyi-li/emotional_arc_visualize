# app.py文件
import io
import numpy as np
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt

# ==============================
# 0. 页面基本配置
# ==============================
st.set_page_config(
    page_title="Emotional Arc 可视化",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Emotional Arc 可视化（基于中文情感 BERT）")
st.write(
    "支持上传 `.txt` 文件或直接输入文本，"
    "对全文做滑动窗口情感分析，并绘制情感弧线。"
)

# ==============================
# 1. 设备 & 模型缓存（只加载一次）
# ==============================
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()
st.sidebar.success(f"模型已加载，设备：{device}")


# ==============================
# 2. 滑动窗口 & 重采样函数
# ==============================
def sliding_windows(text: str, window_size: int = 50, step: int = 40):
    """
    基于字符的滑动窗口。
    window_size: 每个窗口包含的字符数
    step: 每次滑动的步长（字符）
    """
    windows = []
    positions = []   # 每个窗口在原文中的起始字符索引

    n = len(text)
    if n == 0:
        return windows, positions
    if n <= window_size:
        windows.append(text)
        positions.append(0)
        return windows, positions

    for i in range(0, n, step):
        window = text[i:i + window_size]
        if len(window) == 0:
            break
        windows.append(window)
        positions.append(i)
        if len(window) < window_size:  # 触及末尾
            break

    return windows, positions


def sentiment_scores(sent_list, batch_size: int = 32, max_length: int = 64):
    """
    对一批文本批量计算情感得分（正向概率 0-1）
    使用 GPU + batch 推理加速。
    """
    all_scores = []
    if not sent_list:
        return all_scores

    for i in range(0, len(sent_list), batch_size):
        batch = sent_list[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pos_probs = probs[:, 1].detach().cpu().numpy().tolist()

        all_scores.extend(pos_probs)

    return all_scores


def resample_series(values, target_len: int = 20):
    """
    将任意长度的序列线性插值到固定长度 target_len
    返回：(new_values, x_new)
    x_new 是 [0,1] 上的等间隔点
    """
    if target_len <= 0:
        raise ValueError("target_len must be positive")

    if len(values) == 0:
        return [0.0] * target_len, np.linspace(0, 1, target_len).tolist()
    if len(values) == 1:
        return [float(values[0])] * target_len, np.linspace(0, 1, target_len).tolist()

    values = np.array(values, dtype=float)
    n = len(values)

    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    new_values = np.interp(x_new, x_old, values)

    return new_values.tolist(), x_new.tolist()


# ==============================
# 3. 侧边栏参数设置
# ==============================
st.sidebar.header("参数设置")

window_size = st.sidebar.number_input(
    "窗口大小（字符）",
    min_value=10,
    max_value=1000,
    value=50,
    step=5,
)

step_size = st.sidebar.number_input(
    "滑动步长（字符）",
    min_value=1,
    max_value=1000,
    value=40,
    step=1,
)

arc_len = st.sidebar.number_input(
    "重采样点数（情感弧线长度）",
    min_value=5,
    max_value=200,
    value=20,
    step=1,
)

batch_size = st.sidebar.number_input(
    "推理 batch size",
    min_value=1,
    max_value=128,
    value=32,
    step=1,
)

max_length = st.sidebar.number_input(
    "Tokenizer max_length",
    min_value=16,
    max_value=256,
    value=64,
    step=8,
)

st.sidebar.caption("一般保持默认即可，有性能/长度需求再调整。")


# ==============================
# 4. 文本输入区域：上传文件 or 文本框
# ==============================
st.subheader("1️⃣ 输入文本")

col_file, col_text = st.columns(2)

uploaded_file = None
with col_file:
    uploaded_file = st.file_uploader(
        "上传 `.txt` 文件（可选）",
        type=["txt"],
        help="如果选择文件，将优先使用文件内容",
    )

with col_text:
    default_text = ""
    text_input = st.text_area(
        "或者直接在这里输入/粘贴文本",
        value=default_text,
        height=220,
        placeholder="例如：小说片段、长微博、文章内容等……",
    )

# 读取文件内容（若有）
file_text = ""
if uploaded_file is not None:
    # uploaded_file 是一个 BytesIO-like 对象
    bytes_data = uploaded_file.read()
    try:
        file_text = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        # 兜底用 gbk 尝试一下
        try:
            file_text = bytes_data.decode("gbk")
        except UnicodeDecodeError:
            st.error("无法解码该 txt 文件，请确认编码为 UTF-8 或 GBK。")

# 最终使用的文本：优先文件，否则文本框
final_text = file_text.strip() if file_text else text_input.strip()

if not final_text:
    st.info("请上传 txt 文件或在右侧文本框输入内容。")
else:
    st.success(f"当前文本长度：{len(final_text)} 个字符。")


# ==============================
# 5. 情感弧线分析逻辑 + 缓存
# ==============================
@st.cache_data(show_spinner=False)
def compute_emotional_arc(
    text: str,
    window_size: int,
    step_size: int,
    arc_len: int,
    batch_size: int,
    max_length: int,
):
    windows, positions = sliding_windows(text, window_size, step_size)
    scores = sentiment_scores(windows, batch_size=batch_size, max_length=max_length)
    arc_scores, arc_x = resample_series(scores, target_len=arc_len)
    return positions, scores, arc_x, arc_scores, windows


# ==============================
# 6. 点击按钮开始分析
# ==============================
st.subheader("2️⃣ 运行分析")

run_btn = st.button("开始分析 Emotional Arc 🚀", disabled=(not final_text))

if run_btn and final_text:
    with st.spinner("正在进行情感分析（可能需要几秒钟）..."):
        positions, scores, arc_x, arc_scores, windows = compute_emotional_arc(
            final_text,
            window_size=window_size,
            step_size=step_size,
            arc_len=arc_len,
            batch_size=batch_size,
            max_length=max_length,
        )

    # ==========================
    # 6.1 概览信息
    # ==========================
    st.success("分析完成 ✅")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("窗口数量", len(positions))
    with col_b:
        st.metric("原始弧线点数", len(scores))
    with col_c:
        st.metric("重采样点数", len(arc_scores))

    # ==========================
    # 6.2 绘图区域
    # ==========================
    st.subheader("3️⃣ Emotional Arc 可视化")

    col_raw, col_resampled = st.columns(2)

    # 原始情感弧线（未重采样）
    with col_raw:
        st.markdown("**原始情感弧线（按窗口位置）**")
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        if positions and scores:
            ax1.plot(positions, scores, marker="o")
        ax1.set_xlabel("Text Start Position (Character Index)")
        ax1.set_ylabel("Sentiment Score (Positive Prob.)")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    # 重采样后的情感弧线
    with col_resampled:
        st.markdown("**重采样情感弧线（归一化 0–1）**")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        if arc_x and arc_scores:
            ax2.plot(arc_x, arc_scores, marker="o")
        ax2.set_xlabel("Normalized Position (0–1)")
        ax2.set_ylabel("Sentiment Score (Positive Prob.)")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # ==========================
    # 6.3 详细窗口情感表（可选）
    # ==========================
    st.subheader("4️⃣ 详细窗口情感得分（可展开查看）")
    with st.expander("查看每个窗口的文本片段和情感得分"):
        import pandas as pd

        df_rows = []
        for idx, (pos, win, sc) in enumerate(zip(positions, windows, scores)):
            df_rows.append(
                {
                    "窗口序号": idx,
                    "起始位置": pos,
                    "窗口文本": win,
                    "情感得分(正向概率)": sc,
                }
            )
        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True)


# ==============================
# 7. 底部说明
# ==============================
st.markdown("---")
st.caption(
    "模型：IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment；"
    "情感得分越接近 1 表示越正向，越接近 0 越负向。"
)
