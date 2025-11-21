# app.py
import numpy as np
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go


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
    "情感弧线：把文本从头到尾切成许多小片段，分别评估情感（0=负向，1=正向），"
    "按阅读顺序连成一条“情绪轨迹”。将鼠标悬停在任意一点，可以查看对应片段摘要。"
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
# 2. 初始化会话状态（用来保存分析结果）
# ==============================
if "arc_data" not in st.session_state:
    st.session_state.arc_data = None  # 存分析结果


# ==============================
# 3. 滑动窗口 & 重采样函数
# ==============================
def sliding_windows(text: str, window_size: int = 50, step: int = 40):
    """基于字符的滑动窗口。"""
    windows = []
    positions = []

    n = len(text)
    if n == 0:
        return windows, positions
    if n <= window_size:
        windows.append(text)
        positions.append(0)
        return windows, positions

    for i in range(0, n, step):
        window = text[i: i + window_size]
        if not window:
            break
        windows.append(window)
        positions.append(i)
        if len(window) < window_size:
            break

    return windows, positions


def sentiment_scores(sent_list, batch_size: int = 32, max_length: int = 64):
    """对一批文本批量计算情感得分（正向概率 0-1）。"""
    all_scores = []
    if not sent_list:
        return all_scores

    for i in range(0, len(sent_list), batch_size):
        batch = sent_list[i: i + batch_size]
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
    """线性插值到固定长度 target_len（用于对比不同文本）。"""
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
# 4. 侧边栏参数设置
# ==============================
st.sidebar.header("参数设置")

window_size = st.sidebar.number_input(
    "窗口大小（字符）",
    min_value=10,
    max_value=2000,
    value=80,
    step=10,
    help="每次情感分析的字符长度，类似一个“镜头”的大小。",
)

step_size = st.sidebar.number_input(
    "滑动步长（字符）",
    min_value=1,
    max_value=2000,
    value=60,
    step=5,
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
# 5. 文本输入区域：上传文件 or 文本框
# ==============================
st.subheader("1️⃣ 输入文本")

col_file, col_text = st.columns(2)

with col_file:
    uploaded_file = st.file_uploader(
        "上传 `.txt` 文件（可选）",
        type=["txt"],
        help="如果选择文件，将优先使用文件内容。",
    )

with col_text:
    text_input = st.text_area(
        "或者直接在这里输入 / 粘贴文本",
        value="",
        height=220,
        placeholder="例如：小说片段、长微博、文章内容等……",
    )

# 读取文件内容
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

final_text = file_text.strip() if file_text else text_input.strip()

MAX_CHARS = 20000
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
# 6. 情感弧线分析逻辑 + 缓存
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
# 7. 点击按钮开始分析（更新 session_state）
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
        st.session_state.arc_data = None
    else:
        st.session_state.arc_data = {
            "final_text_len": len(final_text),
            "positions": positions,
            "scores": scores,
            "arc_x": arc_x,
            "arc_scores": arc_scores,
            "windows": windows,
        }
        st.success("分析完成 ✅")


# ==============================
# 8. 若已有分析结果，展示交互式情感弧线图（原始 + 重采样）
# ==============================
arc_data = st.session_state.arc_data

if arc_data is not None:
    positions = arc_data["positions"]
    scores = arc_data["scores"]
    arc_x = arc_data["arc_x"]
    arc_scores = arc_data["arc_scores"]
    windows = arc_data["windows"]
    total_len = arc_data["final_text_len"]

    if not positions:
        st.warning("当前没有有效的分析结果，请检查文本或参数后重新分析。")
    else:
        scores_arr = np.array(scores)

        avg_score = float(scores_arr.mean())
        min_score = float(scores_arr.min())
        max_score = float(scores_arr.max())
        min_idx = int(scores_arr.argmin())
        max_idx = int(scores_arr.argmax())
        min_pos = positions[min_idx]
        max_pos = positions[max_idx]

        # ---- 8.1 整体情感概览 ----
        st.subheader("3️⃣ 整体情感概览")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("平均情感得分", f"{avg_score:.3f}")
        with col_b:
            st.metric("最低情感得分", f"{min_score:.3f}", help=f"出现在字符位置约 {min_pos}")
        with col_c:
            st.metric("最高情感得分", f"{max_score:.3f}", help=f"出现在字符位置约 {max_pos}")

        # ---- 8.2 情感弧线图（原始 + 重采样，tab 切换）----
        st.subheader("4️⃣ Emotional Arc 可视化")

        tab_raw, tab_resampled = st.tabs(["原始情感弧线", "重采样情感弧线"])

        # 为 tooltip 准备简短 snippets
        snippets = []
        for w in windows:
            s = w[:50]  # 控制长度，避免 tooltip 太长
            if len(w) > 50:
                s += "..."
            snippets.append(s)

        # --- Tab 1: 原始情感弧线 ---
        with tab_raw:
            st.markdown("**按文本实际位置绘制的情感弧线**（横轴是字符起始位置，纵轴是情感得分）。")

            fig_raw = go.Figure()

            # 主情感弧线
            fig_raw.add_trace(
                go.Scatter(
                    x=positions,
                    y=scores,
                    mode="lines+markers",
                    name="Emotional Arc",
                    line=dict(color="#4F81BD", width=2),
                    marker=dict(color="#4F81BD", size=6),
                    customdata=[[i, snippets[i]] for i in range(len(positions))],
                    hovertemplate=(
                        "<b>Window #%{customdata[0]}</b><br>"
                        "Start: %{x}<br>"
                        "Score: %{y:.3f}<br>"
                        "Snippet: %{customdata[1]}"
                    ),
                )
            )

            # 全局最大 / 最小点（绿色 / 红色）
            fig_raw.add_trace(
                go.Scatter(
                    x=[max_pos],
                    y=[max_score],
                    mode="markers",
                    name="Max score",
                    marker=dict(color="#2E8B57", size=10, symbol="triangle-up"),
                    hovertemplate="Max score<br>Start: %{x}<br>Score: %{y:.3f}",
                )
            )
            fig_raw.add_trace(
                go.Scatter(
                    x=[min_pos],
                    y=[min_score],
                    mode="markers",
                    name="Min score",
                    marker=dict(color="#E24A33", size=10, symbol="triangle-down"),
                    hovertemplate="Min score<br>Start: %{x}<br>Score: %{y:.3f}",
                )
            )

            fig_raw.update_layout(
                template="plotly_white",
                xaxis_title="Text Start Position (Character Index)",
                yaxis_title="Sentiment Score (Positive Prob.)",
                yaxis=dict(range=[0, 1]),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                ),
                margin=dict(l=40, r=20, t=40, b=40),
                hovermode="x unified",
            )

            st.plotly_chart(fig_raw, use_container_width=True)

        # --- Tab 2: 重采样情感弧线 ---
        with tab_resampled:
            st.markdown(
                "**重采样情感弧线**：不看具体字符位置，而是把全文归一化到 0–1，"
                "更像是“从开头到结尾”的整体情绪走势。"
            )

            fig_res = go.Figure()

            if arc_x and arc_scores:
                # 给每个点一个简单的位置标签：开头 / 中段 / 结尾
                customdata_res = []
                for x in arc_x:
                    if x < 0.2:
                        pos_label = "接近开头"
                    elif x > 0.8:
                        pos_label = "接近结尾"
                    else:
                        pos_label = "大概在中间"
                    customdata_res.append(pos_label)

                fig_res.add_trace(
                    go.Scatter(
                        x=arc_x,
                        y=arc_scores,
                        mode="lines+markers",
                        name="Resampled Arc",
                        line=dict(color="#AA6FE8", width=2),
                        marker=dict(color="#AA6FE8", size=6),
                        customdata=customdata_res,
                        hovertemplate=(
                            "位置（0-1）：%{x:.2f}（%{customdata}）<br>"
                            "Score: %{y:.3f}"
                        ),
                    )
                )

            fig_res.update_layout(
                template="plotly_white",
                xaxis_title="Normalized Position (0–1)",
                yaxis_title="Sentiment Score (Positive Prob.)",
                yaxis=dict(range=[0, 1]),
                margin=dict(l=40, r=20, t=40, b=40),
                hovermode="x",
            )

            st.plotly_chart(fig_res, use_container_width=True)

        # ---- 8.3 可选：展开查看完整窗口表格 ----
        with st.expander("📋 展开查看所有窗口的详细得分与文本片段"):
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
# 9. 底部说明
# ==============================
st.markdown("---")
st.caption(
    "模型：IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment；"
    "情感得分越接近 1 表示越正向，越接近 0 越负向。"
    "这是一种自动分析结果，仅供参考和探索文本情绪结构使用。"
)
