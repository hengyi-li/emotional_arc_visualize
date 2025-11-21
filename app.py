# app.py
import numpy as np
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events  # 用于捕获 Plotly 点击事件


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
# 2. 初始化会话状态
# ==============================
if "arc_data" not in st.session_state:
    st.session_state.arc_data = None  # 存分析结果
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0  # 当前选中的窗口索引


# ==============================
# 3. 滑动窗口 & 重采样函数
# ==============================
def sliding_windows(text: str, window_size: int = 50, step: int = 40):
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
# 7. 点击按钮开始分析（只更新 session_state）
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
        scores_arr_tmp = np.array(scores)
        max_idx_tmp = int(scores_arr_tmp.argmax())

        st.session_state.arc_data = {
            "final_text_len": len(final_text),
            "positions": positions,
            "scores": scores,
            "arc_x": arc_x,
            "arc_scores": arc_scores,
            "windows": windows,
        }
        st.session_state.selected_idx = max_idx_tmp  # 默认选中情感最高点

        st.success("分析完成 ✅")


# ==============================
# 8. 若已有分析结果，展示交互式 Emotional Arc
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
        pos_arr = np.array(positions)

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

        # ---- 8.2 布局：左弧线 + 右详情 ----
        st.subheader("4️⃣ 交互式浏览 Emotional Arc")
        col_left, col_right = st.columns([2, 1])

        # 为 tooltip 准备 snippets
        snippets = []
        for w in windows:
            s = w[:50]
            if len(w) > 50:
                s += "..."
            snippets.append(s)

        # 确保 selected_idx 在范围内
        if st.session_state.selected_idx >= len(positions):
            st.session_state.selected_idx = len(positions) - 1
        if st.session_state.selected_idx < 0:
            st.session_state.selected_idx = 0

        # ---- 左侧：弧线 / 重采样 / 表格 ----
        with col_left:
            tab_arc, tab_arc_resampled, tab_table = st.tabs(
                ["原始情感弧线（可点击）", "重采样弧线", "窗口详情表格"]
            )

            # Tab 1: 原始情感弧线（Plotly + 点击交互）
            with tab_arc:
                fig1 = go.Figure()

                # 主线：情感弧线
                fig1.add_trace(
                    go.Scatter(
                        x=positions,
                        y=scores,
                        mode="lines+markers",
                        name="Emotional Arc",
                        customdata=[[i, snippets[i]] for i in range(len(positions))],
                        hovertemplate=(
                            "Window index: %{customdata[0]}<br>"
                            "Start position: %{x}<br>"
                            "Score: %{y:.3f}<br>"
                            "Snippet: %{customdata[1]}"
                        ),
                    )
                )

                # 高亮最高点 & 最低点（全局特征）
                fig1.add_trace(
                    go.Scatter(
                        x=[max_pos],
                        y=[max_score],
                        mode="markers",
                        name="Max score",
                        marker=dict(size=10, symbol="triangle-up"),
                        hovertemplate="Max score<br>Start: %{x}<br>Score: %{y:.3f}",
                    )
                )
                fig1.add_trace(
                    go.Scatter(
                        x=[min_pos],
                        y=[min_score],
                        mode="markers",
                        name="Min score",
                        marker=dict(size=10, symbol="triangle-down"),
                        hovertemplate="Min score<br>Start: %{x}<br>Score: %{y:.3f}",
                    )
                )

                fig1.update_layout(
                    xaxis_title="Text Start Position (Character Index)",
                    yaxis_title="Sentiment Score (Positive Prob.)",
                    yaxis=dict(range=[0, 1]),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                    ),
                    margin=dict(l=40, r=20, t=40, b=40),
                    hovermode="x unified",
                )

                # 用 plotly_events 捕获点击事件
                clicked_points = plotly_events(
                    fig1,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    key="arc_click",
                )

                # 如果点击了某个点，用它来更新当前选中窗口
                if clicked_points:
                    try:
                        new_idx = int(clicked_points[0]["customdata"][0])
                        st.session_state.selected_idx = new_idx
                    except Exception:
                        pass

                # Tab 2: 重采样后的情感弧线
            with tab_arc_resampled:
                fig2 = go.Figure()
                if arc_x and arc_scores:
                    fig2.add_trace(
                        go.Scatter(
                            x=arc_x,
                            y=arc_scores,
                            mode="lines+markers",
                            name="Resampled Arc",
                            hovertemplate="Pos: %{x:.2f}<br>Score: %{y:.3f}",
                        )
                    )

                fig2.update_layout(
                    xaxis_title="Normalized Position (0–1)",
                    yaxis_title="Sentiment Score (Positive Prob.)",
                    yaxis=dict(range=[0, 1]),
                    margin=dict(l=40, r=20, t=40, b=40),
                    hovermode="x",
                )

                st.plotly_chart(fig2, use_container_width=True)

            # Tab 3: 窗口详情表格
            with tab_table:
                st.markdown("**每个窗口的文本片段与情感得分（可排序、筛选）**")
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

        # ---- 右侧：当前窗口详情（由 selected_idx 驱动）----
        with col_right:
            selected_idx = st.session_state.selected_idx
            selected_pos = positions[selected_idx]
            selected_score = scores[selected_idx]
            selected_win = windows[selected_idx]

            center_pos = selected_pos + window_size / 2
            percent = center_pos / max(total_len, 1)

            st.markdown("**当前选中窗口详情**")
            st.markdown(
                f"- 窗口序号：`{selected_idx}` / `{len(positions) - 1}`"
            )
            st.markdown(
                f"- 起始位置：`{selected_pos}` 字符（窗口中心约在全文 `{percent * 100:.1f}%` 处）"
            )
            st.markdown(f"- 情感得分：`{selected_score:.4f}`")

            st.markdown("---")
            st.markdown("**窗口文本内容**")
            st.write(selected_win)

            st.markdown("---")
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("⬅ 上一窗口", disabled=(selected_idx <= 0)):
                    st.session_state.selected_idx = max(0, selected_idx - 1)
            with col_next:
                if st.button("下一窗口 ➡", disabled=(selected_idx >= len(positions) - 1)):
                    st.session_state.selected_idx = min(
                        len(positions) - 1, selected_idx + 1
                    )

            st.caption(
                "交互说明：可以**点击左侧情感弧线上任意一点**，右侧会显示对应窗口的文本；"
                "也可以使用“上一窗口 / 下一窗口”按钮逐步浏览。"
            )

# ==============================
# 9. 底部说明
# ==============================
st.markdown("---")
st.caption(
    "模型：IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment；"
    "情感得分越接近 1 表示越正向，越接近 0 越负向。"
    "这是一种自动分析结果，仅供参考和探索文本情绪结构使用。"
)
