# app.py
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
    "支持上传 `.txt` 文件或直接输入文本，对全文做滑动窗口情感分析，"
    "展示故事在阅读过程中的情绪起伏曲线（Emotional Arc）。"
)

st.info(
    "情感弧线 = 文本从头到尾，情绪如何在“时间维度”上起伏的一条曲线。"
    "曲线越往上，表示越偏正向；越往下，表示越偏负向。"
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
    positions = []  # 每个窗口在原文中的起始字符索引

    n = len(text)
    if n == 0:
        return windows, positions
    if n <= window_size:
        windows.append(text)
        positions.append(0)
        return windows, positions

    for i in range(0, n, step):
        window = text[i : i + window_size]
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
        batch = sent_list[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
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
    max_value=2000,
    value=50,
    step=5,
    help="每次情感分析的字符长度，类似一个“片段”的大小。",
)

step_size = st.sidebar.number_input(
    "滑动步长（字符）",
    min_value=1,
    max_value=2000,
    value=40,
    step=1,
    help="窗口每次向前滑动的字符数。步长越小，曲线越平滑，但计算越慢。",
)

arc_len = st.sidebar.number_input(
    "弧线点数（重采样后）",
    min_value=5,
    max_value=200,
    value=20,
    step=1,
    help="将整条情感弧线压缩到固定数量的点，方便对比不同文本。",
)

st.sidebar.markdown("---")
advanced = st.sidebar.checkbox("显示高级参数", value=False)

if advanced:
    batch_size = st.sidebar.number_input(
        "推理 batch size",
        min_value=1,
        max_value=128,
        value=32,
        step=1,
        help="一次送入模型的窗口数量。过大可能导致内存不足。",
    )
    max_length = st.sidebar.number_input(
        "Tokenizer max_length",
        min_value=16,
        max_value=256,
        value=64,
        step=8,
        help="单个窗口的最大 token 长度，通常保持默认即可。",
    )
else:
    batch_size = 32
    max_length = 64
    st.sidebar.caption("高级参数使用默认设置。如需性能调优可勾选上方开关。")


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
        help="如果选择文件，将优先使用文件内容。",
    )

with col_text:
    default_text = ""
    text_input = st.text_area(
        "或者直接在这里输入 / 粘贴文本",
        value=default_text,
        height=220,
        placeholder="例如：小说片段、长微博、文章内容等……",
    )

# 读取文件内容（若有）
file_text = ""
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    try:
        file_text = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            file_text = bytes_data.decode("gbk")
        except UnicodeDecodeError:
            st.error("无法解码该 txt 文件，请确认编码为 UTF-8 或 GBK。")

# 最终使用的文本：优先文件，否则文本框
final_text = file_text.strip() if file_text else text_input.strip()

MAX_CHARS = 20000  # 建议上限
if not final_text:
    st.info("请上传 txt 文件或在右侧文本框输入内容。")
else:
    if file_text:
        st.success(
            f"已使用上传文件内容：**{uploaded_file.name}**，长度 {len(final_text)} 个字符。"
        )
    else:
        st.success(f"已使用文本框内容，长度 {len(final_text)} 个字符。")

    if len(final_text) > MAX_CHARS:
        st.warning(
            f"当前文本长度为 {len(final_text)} 个字符，超过推荐上限 {MAX_CHARS}。"
            "分析可能较慢，建议截取关键片段或章节试试看。"
        )


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

    if not positions:
        st.warning("未生成任何窗口，可能是参数设置不合理（例如窗口太大、文本太短）。")
    else:
        # ==========================
        # 6.1 概览信息
        # ==========================
        st.success("分析完成 ✅")

        scores_arr = np.array(scores)
        avg_score = float(scores_arr.mean())
        min_score = float(scores_arr.min())
        max_score = float(scores_arr.max())
        min_idx = int(scores_arr.argmin())
        max_idx = int(scores_arr.argmax())
        min_pos = positions[min_idx]
        max_pos = positions[max_idx]

        st.subheader("3️⃣ 整体情感概览")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("平均情感得分", f"{avg_score:.3f}")
        with col_b:
            st.metric("最低情感得分", f"{min_score:.3f}", help=f"出现在字符位置约 {min_pos}")
        with col_c:
            st.metric("最高情感得分", f"{max_score:.3f}", help=f"出现在字符位置约 {max_pos}")

        # ==========================
        # 6.2 结果展示：Tabs
        # ==========================
        st.subheader("4️⃣ Emotional Arc 详细结果")
        tab_arc, tab_arc_resampled, tab_table = st.tabs(
            ["原始情感弧线", "重采样弧线", "窗口详情"]
        )

        # ---- Tab 1: 原始情感弧线 ----
        with tab_arc:
            st.markdown("**原始情感弧线（按窗口起始位置）**")
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            if positions and scores:
                ax1.plot(positions, scores, marker="o")

                # 标记最高 & 最低点
                pos_arr = np.array(positions)
                ax1.scatter(
                    [pos_arr[max_idx]],
                    [scores_arr[max_idx]],
                    s=60,
                    edgecolors="black",
                    facecolors="none",
                    linewidths=1.5,
                )
                ax1.scatter(
                    [pos_arr[min_idx]],
                    [scores_arr[min_idx]],
                    s=60,
                    edgecolors="black",
                    facecolors="none",
                    linewidths=1.5,
                )

            ax1.set_xlabel("Text Start Position (Character Index)")
            ax1.set_ylabel("Sentiment Score (Positive Prob.)")
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        # ---- Tab 2: 重采样后的情感弧线 ----
        with tab_arc_resampled:
            st.markdown("**重采样情感弧线（归一化位置 0–1）**")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            if arc_x and arc_scores:
                ax2.plot(arc_x, arc_scores, marker="o")
            ax2.set_xlabel("Normalized Position (0–1)")
            ax2.set_ylabel("Sentiment Score (Positive Prob.)")
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        # ---- Tab 3: 窗口详情表格 ----
        with tab_table:
            st.markdown("**每个窗口的文本片段与情感得分**")
            import pandas as pd

            df_rows = []
            for idx, (pos, win, sc) in enumerate(zip(positions, windows, scores)):
                df_rows.append(
                    {
                        "窗口序号": idx,
                        "起始位置（字符索引）": pos,
                        "窗口文本": win,
                        "情感得分 (Positive Prob.)": sc,
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
    "这是一种自动分析结果，仅供参考和探索文本情绪结构使用。"
)
